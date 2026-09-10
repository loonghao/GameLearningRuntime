extends SceneTree
# Source-integrated Node3D sample over GLR stdio. Decisions move a transform;
# no physics frames, authentication or external process binding are claimed.
var target := Node3D.new()
var episode := ""
var cursor := 0

func tensor(dtype: String, bytes: PackedByteArray) -> Dictionary:
	return {"shape": [1], "dtype": dtype, "data": Marshalls.raw_to_base64(bytes)}

func float_tensor(value: float) -> Dictionary:
	var bytes := PackedByteArray()
	bytes.resize(4)
	bytes.encode_float(0, value)
	return tensor("float32", bytes)

func spec(path: String, dtype: String, kind: String) -> Dictionary:
	return {"path": path, "shape": [1], "dtype": dtype, "kind": kind}

func describe() -> Dictionary:
	var action := spec("move", "int32", "discrete")
	action.merge({"minimum": -1, "maximum": 1})
	return {"environment_id": "glr.godot.node-v1", "protocol_version": "1.0",
		"observations": [spec("position", "float32", "continuous")],
		"actions": [action], "action_masks": [],
		"reward": spec("reward", "float32", "continuous"),
		"done": spec("done", "bool", "binary"), "metadata": {},
		"capabilities": ["host-stdio", "reset", "live-attach", "step", "native-action", "semantic-observation"]}

func state() -> Dictionary:
	var done := absf(target.position.x) >= 3
	return {"episode_id": episode, "step_id": cursor,
		"timestamp_ns": int(Time.get_unix_time_from_system() * 1000000000),
		"observation": {"position": float_tensor(target.position.x)},
		"reward": float_tensor(1.0 if done else 0.0),
		"terminated": tensor("bool", PackedByteArray([1 if done else 0])),
		"truncated": tensor("bool", PackedByteArray([0])),
		"events": [], "info": {}}

func refuse(message: String) -> Dictionary:
	return {"error": {"code": "contract", "message": message, "retryable": false}}

func dispatch(operation: String, payload: Dictionary) -> Dictionary:
	if operation == "describe":
		return {"result": describe()}
	if operation == "close":
		return {"result": {"closed": true}}
	if operation == "reset" or operation == "attach":
		if not payload.get("options", {}).is_empty():
			return refuse("Unknown lifecycle options")
		if operation == "attach" and absf(target.position.x) >= 3:
			return refuse("Reset the terminal sample first")
		if operation == "reset":
			target.position = Vector3.ZERO
		var hex := Crypto.new().generate_random_bytes(16).hex_encode()
		episode = "%s-%s-%s-%s-%s" % [hex.substr(0, 8), hex.substr(8, 4), hex.substr(12, 4), hex.substr(16, 4), hex.substr(20, 12)]
		cursor = 0
		return {"result": state()}
	if operation != "step":
		return refuse("Unsupported operation")
	if episode.is_empty() or payload.get("episode_id") != episode or payload.get("expected_step_id") != cursor + 1:
		return refuse("Stale episode or step")
	if absf(target.position.x) >= 3:
		return refuse("Episode is done")
	for key in payload:
		if key not in ["episode_id", "expected_step_id", "action"]:
			return refuse("Unsupported step option")
	var actions = payload.get("action", {})
	if not actions is Dictionary or actions.size() != 1 or not actions.has("move"):
		return refuse("Expected move tensor")
	var move = actions["move"]
	if not move is Dictionary or move.get("dtype") != "int32" or not move.get("data") is String:
		return refuse("Expected int32 move tensor")
	var shape = move.get("shape")
	if not shape is Array or shape.size() != 1 or shape[0] != 1:
		return refuse("Expected one move value")
	var bytes := Marshalls.base64_to_raw(move["data"])
	if bytes.size() != 4:
		return refuse("Invalid move bytes")
	var value := bytes.decode_s32(0)
	if value < -1 or value > 1:
		return refuse("Move outside bounds")
	target.position.x += value
	cursor += 1
	return {"result": state()}

func _initialize() -> void:
	root.add_child(target)
	while true:
		var line := OS.read_string_from_stdin()
		if line.is_empty():
			break
		if line.length() > 1048576:
			break
		var request = JSON.parse_string(line)
		if not request is Dictionary or request.get("schema") != "glr.host.v1" or not request.get("payload") is Dictionary:
			break
		var operation = request.get("operation", "")
		if not operation is String:
			break
		var response := dispatch(operation, request["payload"])
		response.merge({"schema": "glr.host.v1", "request_id": request.get("request_id"), "ok": response.has("result")})
		print(JSON.stringify(response))
		if operation == "close":
			break
	quit(0)
