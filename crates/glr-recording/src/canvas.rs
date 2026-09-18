//! Frame geometry and canvas compositing shared by every platform.
//!
//! The encoded stream keeps one resolution for a whole session: a window that
//! resizes (or exceeds `max_width`) is letterboxed or downscaled into the canvas
//! chosen from the first frame instead of changing the stream mid recording.

/// Bytes per pixel of the BGRA canvases exchanged with the encoder.
pub const BYTES_PER_PIXEL: usize = 4;

/// Resolution of one encoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameGeometry {
    /// Width in pixels, always even.
    pub width: u32,
    /// Height in pixels, always even.
    pub height: u32,
}

impl FrameGeometry {
    /// Size of the BGRA buffer backing this geometry.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        self.width as usize * self.height as usize * BYTES_PER_PIXEL
    }
}

/// Rounds a dimension down to the even value H.264 requires, with a floor of two.
#[must_use]
pub const fn align_even(value: u32) -> u32 {
    if value < 2 { 2 } else { value & !1 }
}

/// Chooses the session resolution for a source frame.
///
/// The long edge is capped at `max_width` (when non-zero), the aspect ratio is
/// preserved and both dimensions are even-aligned.
#[must_use]
pub fn plan_size(width: u32, height: u32, max_width: u32) -> FrameGeometry {
    let width = width.max(2);
    let height = height.max(2);
    let long_edge = width.max(height);
    let scale = if max_width > 0 && long_edge > max_width {
        f64::from(max_width) / f64::from(long_edge)
    } else {
        1.0
    };
    FrameGeometry {
        width: align_even((f64::from(width) * scale).round() as u32),
        height: align_even((f64::from(height) * scale).round() as u32),
    }
}

/// Composites a top-down BGRA source into a bottom-up BGRA destination canvas.
///
/// Windows expects raw sample buffers bottom-up, so row `0` of `destination` is
/// the bottom row of the image. The source is centred; when it does not fit it
/// is box-downscaled (never upscaled) and everything else stays black.
pub fn composite_bgra(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    destination: &mut [u8],
    destination_width: u32,
    destination_height: u32,
) {
    if source_width == 0 || source_height == 0 || destination_width == 0 || destination_height == 0 {
        return;
    }
    let required = source_width as usize * source_height as usize * BYTES_PER_PIXEL;
    if source.len() < required {
        return;
    }
    destination.fill(0);

    let fit_width = source_width.min(destination_width);
    let fit_height = source_height.min(destination_height);
    // Preserve the aspect ratio while fitting, and never upscale.
    let scale = (f64::from(destination_width) / f64::from(source_width))
        .min(f64::from(destination_height) / f64::from(source_height))
        .min(1.0);
    let scaled_width = ((f64::from(source_width) * scale).round() as u32)
        .clamp(1, fit_width)
        .max(1);
    let scaled_height = ((f64::from(source_height) * scale).round() as u32)
        .clamp(1, fit_height)
        .max(1);
    let offset_x = (destination_width - scaled_width) / 2;
    let offset_y = (destination_height - scaled_height) / 2;

    for row in 0..scaled_height {
        let source_row_start = (row as usize * source_height as usize) / scaled_height as usize;
        let source_row_end = (((row as usize + 1) * source_height as usize) / scaled_height as usize)
            .max(source_row_start + 1)
            .min(source_height as usize);
        let destination_row = destination_height - 1 - (offset_y + row);
        for column in 0..scaled_width {
            let source_column_start = (column as usize * source_width as usize) / scaled_width as usize;
            let source_column_end = (((column as usize + 1) * source_width as usize)
                / scaled_width as usize)
                .max(source_column_start + 1)
                .min(source_width as usize);
            let (mut blue, mut green, mut red, mut samples) = (0u32, 0u32, 0u32, 0u32);
            for source_row in source_row_start..source_row_end {
                for source_column in source_column_start..source_column_end {
                    let index = (source_row * source_width as usize + source_column) * BYTES_PER_PIXEL;
                    blue += u32::from(source[index]);
                    green += u32::from(source[index + 1]);
                    red += u32::from(source[index + 2]);
                    samples += 1;
                }
            }
            let index =
                (destination_row as usize * destination_width as usize + offset_x as usize + column as usize)
                    * BYTES_PER_PIXEL;
            destination[index] = (blue / samples) as u8;
            destination[index + 1] = (green / samples) as u8;
            destination[index + 2] = (red / samples) as u8;
            destination[index + 3] = 0xFF;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BYTES_PER_PIXEL, FrameGeometry, align_even, composite_bgra, plan_size};

    fn pixel(buffer: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
        let index = (y as usize * width as usize + x as usize) * BYTES_PER_PIXEL;
        [buffer[index], buffer[index + 1], buffer[index + 2], buffer[index + 3]]
    }

    fn source(width: u32, height: u32) -> Vec<u8> {
        let mut buffer = vec![0u8; width as usize * height as usize * BYTES_PER_PIXEL];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let index = (y * width as usize + x) * BYTES_PER_PIXEL;
                buffer[index] = (x + 1) as u8;
                buffer[index + 1] = (y + 1) as u8;
                buffer[index + 2] = 0x80;
                buffer[index + 3] = 0xFF;
            }
        }
        buffer
    }

    #[test]
    fn align_even_keeps_dimensions_encodable() {
        assert_eq!(align_even(0), 2);
        assert_eq!(align_even(1), 2);
        assert_eq!(align_even(2), 2);
        assert_eq!(align_even(3), 2);
        assert_eq!(align_even(4), 4);
        assert_eq!(align_even(1921), 1920);
    }

    #[test]
    fn plan_size_caps_the_long_edge_and_keeps_aspect() {
        let geometry = plan_size(3840, 2160, 1920);
        assert_eq!(geometry, FrameGeometry { width: 1920, height: 1080 });
        assert_eq!(plan_size(800, 600, 1920), FrameGeometry { width: 800, height: 600 });
        assert_eq!(plan_size(800, 600, 0), FrameGeometry { width: 800, height: 600 });
    }

    #[test]
    fn plan_size_even_aligns_odd_sources() {
        let geometry = plan_size(1281, 721, 0);
        assert_eq!(geometry.width % 2, 0);
        assert_eq!(geometry.width, 1280);
        assert_eq!(geometry.height, 720);
    }

    #[test]
    fn composite_flips_rows_for_windows_samples() {
        let source = source(2, 2);
        let mut destination = vec![0u8; 2 * 2 * BYTES_PER_PIXEL];
        composite_bgra(&source, 2, 2, &mut destination, 2, 2);
        // Top-down row 0 of the source becomes the bottom-up row 1.
        assert_eq!(pixel(&destination, 2, 0, 1), [1, 1, 0x80, 0xFF]);
        assert_eq!(pixel(&destination, 2, 1, 1), [2, 1, 0x80, 0xFF]);
        assert_eq!(pixel(&destination, 2, 0, 0), [1, 2, 0x80, 0xFF]);
    }

    #[test]
    fn composite_letterboxes_a_smaller_frame() {
        let source = source(2, 2);
        let mut destination = vec![0u8; 4 * 4 * BYTES_PER_PIXEL];
        composite_bgra(&source, 2, 2, &mut destination, 4, 4);
        assert_eq!(pixel(&destination, 4, 0, 0), [0, 0, 0, 0]);
        // Bottom-up row 2 is top-down row 1, the centred source's first row.
        assert_eq!(pixel(&destination, 4, 1, 2), [1, 1, 0x80, 0xFF]);
        assert_eq!(pixel(&destination, 4, 2, 2), [2, 1, 0x80, 0xFF]);
        assert_eq!(pixel(&destination, 4, 3, 3), [0, 0, 0, 0]);
    }

    #[test]
    fn composite_downscales_with_a_box_filter() {
        let source = source(4, 4);
        let mut destination = vec![0u8; 2 * 2 * BYTES_PER_PIXEL];
        composite_bgra(&source, 4, 4, &mut destination, 2, 2);
        // Each destination pixel averages a 2x2 block: columns 1..2 average to 1.5 -> 1.
        assert_eq!(pixel(&destination, 2, 0, 1), [1, 1, 0x80, 0xFF]);
        assert_eq!(pixel(&destination, 2, 1, 0), [3, 3, 0x80, 0xFF]);
    }

    #[test]
    fn composite_ignores_degenerate_input() {
        let mut destination = vec![0u8; 16];
        composite_bgra(&[], 0, 0, &mut destination, 2, 2);
        assert!(destination.iter().all(|value| *value == 0));
        let source = source(2, 2);
        composite_bgra(&source, 2, 2, &mut destination, 0, 0);
        assert!(destination.iter().all(|value| *value == 0));
    }
}
