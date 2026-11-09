use std::slice;

#[no_mangle]
pub extern "C" fn rust_engine_version() -> i32 {
    100
}

#[no_mangle]
pub extern "C" fn rust_red_compensate(
    data: *mut u8,
    len: usize,
    red_gain: f32,
) -> i32 {
    if data.is_null() || len == 0 || len % 3 != 0 {
        return -1;
    }

    if (red_gain - 1.0).abs() < 0.01 {
        return 0;
    }

    let mut lut = [0u8; 256];
    for i in 0..256 {
        let val = (i as f32 * red_gain).round();
        lut[i] = if val > 255.0 { 255 } else if val < 0.0 { 0 } else { val as u8 };
    }

    let pixels = unsafe { slice::from_raw_parts_mut(data, len) };

    for chunk in pixels.chunks_exact_mut(3) {
        chunk[2] = lut[chunk[2] as usize];
    }

    0
}
#[no_mangle]
pub extern "C" fn rust_color_balance(
    data: *mut u8,
    len: usize,
    clip_percentile: f32,
) -> i32 {
    if data.is_null() || len == 0 || len % 3 != 0 {
        return -1;
    }

    let num_pixels = len / 3;
    if num_pixels == 0 {
        return 0;
    }

    let p = clip_percentile.clamp(0.01, 10.0) / 100.0;
    let low_count = (num_pixels as f32 * p).round() as usize;
    let high_count = (num_pixels as f32 * (1.0 - p)).round() as usize;

    let pixels = unsafe { slice::from_raw_parts_mut(data, len) };

    let mut hist_b = [0usize; 256];
    let mut hist_g = [0usize; 256];
    let mut hist_r = [0usize; 256];

    for chunk in pixels.chunks_exact(3) {
        hist_b[chunk[0] as usize] += 1;
        hist_g[chunk[1] as usize] += 1;
        hist_r[chunk[2] as usize] += 1;
    }

    let build_lut = |hist: &[usize; 256]| -> [u8; 256] {
        let mut cumsum = 0usize;
        let mut low_val = 0usize;
        let mut high_val = 255usize;

        let mut found_low = false;
        for i in 0..256 {
            cumsum += hist[i];
            if !found_low && cumsum >= low_count {
                low_val = i;
                found_low = true;
            }
            if cumsum >= high_count {
                high_val = i;
                break;
            }
        }

        let mut lut = [0u8; 256];
        if high_val > low_val {
            let scale = 255.0 / (high_val - low_val) as f32;
            for i in 0..256 {
                if i <= low_val {
                    lut[i] = 0;
                } else if i >= high_val {
                    lut[i] = 255;
                } else {
                    lut[i] = ((i - low_val) as f32 * scale).round() as u8;
                }
            }
        } else {
            for i in 0..256 {
                lut[i] = i as u8;
            }
        }
        lut
    };

    let lut_b = build_lut(&hist_b);
    let lut_g = build_lut(&hist_g);
    let lut_r = build_lut(&hist_r);

    for chunk in pixels.chunks_exact_mut(3) {
        chunk[0] = lut_b[chunk[0] as usize];
        chunk[1] = lut_g[chunk[1] as usize];
        chunk[2] = lut_r[chunk[2] as usize];
    }

    0
}
