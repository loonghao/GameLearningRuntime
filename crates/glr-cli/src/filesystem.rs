#[cfg(unix)]
use crate::error::Error;
use crate::error::Result;
use std::path::Path;

// Atomic no-replace promotion. A racing destination must never be overwritten.
pub(crate) fn promote(source: &Path, destination: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
        let destination: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        if unsafe {
            windows_sys::Win32::Storage::FileSystem::MoveFileW(
                source.as_ptr(),
                destination.as_ptr(),
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(source.as_os_str().as_bytes())
            .map_err(|_| Error::Invalid("invalid destination".into()))?;
        let destination = std::ffi::CString::new(destination.as_os_str().as_bytes())
            .map_err(|_| Error::Invalid("invalid destination".into()))?;
        #[cfg(target_os = "linux")]
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                destination.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        #[cfg(target_os = "macos")]
        let result =
            unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
        if result != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}
