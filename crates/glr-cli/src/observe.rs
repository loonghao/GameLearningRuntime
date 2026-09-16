//! Embedded Axum transport; all blocking storage work stays off its reactor.
use crate::dashboard::Dashboard;
use crate::error::{Error, Result};
use crate::observation::{Observation, SCHEMA};
use crate::project::Project;
use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub struct Observer {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    pub url: String,
}
impl Drop for Observer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if self.worker.as_ref().is_some_and(|w| w.is_finished()) {
            let _ = self.worker.take().unwrap().join();
        }
    }
}
#[derive(Clone)]
struct WebState {
    observation: Arc<Observation>,
    dashboard: Option<Arc<Dashboard>>,
    port: u16,
    permits: Arc<tokio::sync::Semaphore>,
}

pub fn start_default(project: &Project, enabled: bool) -> Option<Observer> {
    if !enabled {
        return None;
    }
    match start(project, 7432, false).or_else(|_| start(project, 0, false)) {
        Ok(observer) => {
            eprintln!(
                "GLR observation: {} (stops with this command; use glr dashboard for persistent controls)",
                observer.url
            );
            Some(observer)
        }
        Err(error) => {
            eprintln!("GLR observation unavailable: {error}");
            None
        }
    }
}
pub fn serve(project: &Project, port: u16, as_json: bool, controls: bool) -> Result<i32> {
    let mut observer = start(project, port, controls)?;
    let ready = json!({"schema_version": crate::commands::CLI_OUTPUT_SCHEMA_VERSION,
        "command": if controls {"dashboard.ready"} else {"observe.ready"}, "data": {"schema_version": SCHEMA, "url": observer.url,
        "read_only": !controls, "environment_id": project.environment_id}});
    if as_json {
        println!("{}", ready);
    } else {
        println!("GLR dashboard: {}\nPress Ctrl+C to stop.", observer.url);
    }
    if let Some(worker) = observer.worker.take() {
        worker
            .join()
            .map_err(|_| Error::Contract("dashboard worker stopped unexpectedly".into()))?;
    }
    Ok(0)
}
fn start(project: &Project, port: u16, controls: bool) -> Result<Observer> {
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();
    let url = format!("http://127.0.0.1:{port}/");
    let stop = Arc::new(AtomicBool::new(false));
    let shutdown = stop.clone();
    let state = WebState {
        observation: Arc::new(Observation {
            data_dir: project.data_dir.clone(),
            environment_id: project.environment_id.clone(),
        }),
        dashboard: if controls {
            let token = std::env::var("GLR_TELEMETRY_TOKEN").unwrap_or_else(|_| {
                format!(
                    "{}{}",
                    uuid::Uuid::new_v4().simple(),
                    uuid::Uuid::new_v4().simple()
                )
            });
            if !(32..=128).contains(&token.len())
                || !token
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            {
                return Err(Error::Invalid(
                    "GLR_TELEMETRY_TOKEN must contain 32..128 ASCII letters, digits, _, . or -"
                        .into(),
                ));
            }
            Some(Arc::new(
                Dashboard::new(project)?.with_telemetry(format!("{url}api/v1/telemetry"), token),
            ))
        } else {
            None
        },
        port,
        permits: Arc::new(tokio::sync::Semaphore::new(8)),
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let worker = thread::Builder::new()
        .name("glr-dashboard".into())
        .spawn(move || {
            runtime.block_on(async move {
                let listener = match tokio::net::TcpListener::from_std(listener) {
                    Ok(l) => l,
                    Err(_) => return,
                };
                let app = Router::new()
                    .fallback(handle)
                    .layer(DefaultBodyLimit::max(65536))
                    .with_state(state);
                let _ = axum::serve(listener, app)
                    .with_graceful_shutdown(async move {
                        while !shutdown.load(Ordering::Relaxed) {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    })
                    .await;
            });
        })?;
    Ok(Observer {
        stop,
        worker: Some(worker),
        url,
    })
}
async fn handle(
    State(state): State<WebState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let header = |name: &str| headers.get(name).and_then(|h| h.to_str().ok());
    let allowed = [
        format!("127.0.0.1:{}", state.port),
        format!("localhost:{}", state.port),
    ];
    let host_ok = header("host").is_some_and(|h| allowed.iter().any(|a| a == h));
    let origin_ok =
        header("origin").is_none_or(|o| allowed.iter().any(|a| o == format!("http://{a}")));
    let site_ok = header("sec-fetch-site").is_none_or(|s| s == "same-origin" || s == "none");
    let ingest = method == Method::POST && uri.path() == "/api/v1/telemetry";
    let mut result = if !host_ok || !origin_ok || !site_ok {
        (
            403,
            "application/json",
            b"{\"error\":\"loopback same-origin requests only\"}".to_vec(),
        )
    } else if method != Method::GET && (method != Method::POST || state.dashboard.is_none()) {
        (
            405,
            "application/json",
            b"{\"error\":\"method unavailable\"}".to_vec(),
        )
    } else if ingest
        && !state
            .dashboard
            .as_ref()
            .is_some_and(|d| d.telemetry_authorized(header("authorization")))
    {
        (
            401,
            "application/json",
            b"{\"error\":\"bridge telemetry requires a bearer token\"}".to_vec(),
        )
    } else if method == Method::POST
        && ((!ingest && header("origin").is_none())
            || !header("content-type").is_some_and(|v| {
                v.split(';')
                    .next()
                    .is_some_and(|mime| mime.trim() == "application/json")
            }))
    {
        (
            403,
            "application/json",
            b"{\"error\":\"mutations require same-origin JSON\"}".to_vec(),
        )
    } else {
        let permit = match state.permits.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                return (StatusCode::TOO_MANY_REQUESTS, "dashboard busy; retry").into_response();
            }
        };
        let path = uri.to_string();
        let state = state.clone();
        let task = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            if ingest {
                let receipt = crate::telemetry::ingest(
                    &state.observation.data_dir,
                    &state.observation.environment_id,
                    &body,
                )?;
                return Ok((
                    200,
                    "application/json; charset=utf-8",
                    serde_json::to_vec(&receipt)?,
                ));
            }
            if path.starts_with("/api/v1/control/") {
                if let Some(dashboard) = state.dashboard {
                    let value = dashboard.route(method.as_str(), &path, &body)?;
                    return Ok((
                        200,
                        "application/json; charset=utf-8",
                        serde_json::to_vec(&value)?,
                    ));
                }
                return Ok((
                    403,
                    "application/json",
                    b"{\"error\":\"observation mode is read-only\"}".to_vec(),
                ));
            }
            if method != Method::GET {
                return Ok((
                    405,
                    "application/json",
                    b"{\"error\":\"read-only endpoint\"}".to_vec(),
                ));
            }
            if path.split('?').next() == Some("/api/v1/health") {
                return Ok((
                    200,
                    "application/json",
                    serde_json::to_vec(
                        &json!({"schema_version":SCHEMA,"version":env!("CARGO_PKG_VERSION"),"read_only":state.dashboard.is_none(),"environment_id":state.observation.environment_id}),
                    )?,
                ));
            }
            route(&path, &state.observation)
        });
        match task.await {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => (400, "application/json", json!({"schema_version":SCHEMA,"error":{"type":error.kind(),"message":error.to_string()}}).to_string().into_bytes()),
            Err(_) => (500, "application/json", b"{\"error\":\"dashboard worker failed\"}".to_vec()),
        }
    };
    if result.2.len() > 8 * 1024 * 1024 {
        result = (
            413,
            "application/json",
            b"{\"error\":\"response exceeds 8 MiB\"}".to_vec(),
        );
    }
    let (mut status, content_type, mut bytes) = result;
    let etag = format!("\"{:x}\"", Sha256::digest(&bytes));
    if status == 200 && header("if-none-match") == Some(etag.as_str()) {
        status = 304;
        bytes.clear();
    }
    let mut response = (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        bytes,
    )
        .into_response();
    for (name, value) in [
        ("content-type", content_type),
        ("cache-control", "no-cache"),
        ("etag", etag.as_str()),
        ("x-content-type-options", "nosniff"),
        ("referrer-policy", "no-referrer"),
        (
            "content-security-policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        ),
    ] {
        response.headers_mut().insert(
            axum::http::header::HeaderName::from_static(name),
            value.parse().expect("valid response header"),
        );
    }
    response
}

type WebResult = (u16, &'static str, Vec<u8>);

fn route(url: &str, observation: &Observation) -> Result<WebResult> {
    if url.len() > 4096 {
        return Err(Error::Invalid("request URL too long".into()));
    }
    let url = reqwest::Url::parse(&format!("http://localhost{url}"))
        .map_err(|_| Error::Invalid("invalid URL".into()))?;
    let mut query = HashMap::new();
    for (key, value) in url.query_pairs() {
        if query.insert(key.into_owned(), value.into_owned()).is_some() {
            return Err(Error::Invalid("duplicate query parameter".into()));
        }
    }
    let field = |name: &str| {
        query
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| Error::Invalid(format!("missing {name}")))
    };
    let number = |name: &str, default: i64| -> Result<i64> {
        query.get(name).map_or(Ok(default), |text| {
            text.parse::<i64>()
                .ok()
                .filter(|n| *n >= if name == "events_after" { -1 } else { 0 })
                .ok_or_else(|| Error::Invalid(format!("{name} has an invalid integer cursor")))
        })
    };
    let data: Value = match url.path() {
        "/api/v1/telemetry/schema" => crate::telemetry::schema(),
        "/api/v1/telemetry/state" => crate::telemetry::latest(
            &observation.data_dir,
            &observation.environment_id,
            field("run")?,
        )?,
        "/" => {
            return Ok((
                200,
                "text/html; charset=utf-8",
                include_bytes!("web/index.html").to_vec(),
            ));
        }
        "/app.js" => {
            return Ok((
                200,
                "text/javascript; charset=utf-8",
                include_bytes!("web/app.js").to_vec(),
            ));
        }
        "/controls.js" => {
            return Ok((
                200,
                "text/javascript; charset=utf-8",
                include_bytes!("web/controls.js").to_vec(),
            ));
        }
        "/style.css" => {
            return Ok((
                200,
                "text/css; charset=utf-8",
                include_bytes!("web/style.css").to_vec(),
            ));
        }
        "/api/v1/health" => {
            json!({"schema_version": SCHEMA, "version": env!("CARGO_PKG_VERSION"), "environment_id": observation.environment_id, "read_only": true})
        }
        "/api/v1/runs" => observation.runs(query.get("before").map(String::as_str))?,
        "/api/v1/snapshot" => {
            let limit = number("limit", 250)?;
            if !(1..=250).contains(&limit) {
                return Err(Error::Invalid("limit must be 1..250".into()));
            }
            observation.snapshot(
                field("run")?,
                number("events_after", -1)?,
                number("metrics_after", 0)?,
                limit as u32,
            )?
        }
        "/api/v1/log" => observation.log(
            field("run")?,
            field("path")?,
            query
                .get("offset")
                .map(|_| number("offset", 0).map(|n| n as u64))
                .transpose()?,
        )?,
        _ => {
            return Ok((
                404,
                "application/json",
                b"{\"error\":\"unknown observation endpoint\"}".to_vec(),
            ));
        }
    };
    Ok((
        200,
        "application/json; charset=utf-8",
        serde_json::to_vec(&data)?,
    ))
}
