use std::{
    fs::File,
    io::{BufWriter, Write},
};

use desperado::{Gain, IqAsyncSource};
use num_complex::Complex;
use tokio::runtime::Runtime;
#[cfg(not(feature = "post-process"))]
use {futures::stream::StreamExt, rustfft::FftPlanner};

#[cfg(feature = "post-process")]
mod post_process;

// Total duration in seconds.
const DURATION: f32 = 3600.0;
// Per-channel stats window size in seconds.
const WINDOW: f32 = 0.1;

// SDR config for BLE lower half of frequency band (2.4 GHz - 2.44 GHz).
const SDR_CHANNEL: usize = 0;
const CENTER_FREQ: u32 = 2_420_000_000;
const SAMPLE_RATE: u32 = 40_000_000;
const GAIN: f64 = 50.0;
const BANDWIDTH: u32 = SAMPLE_RATE;

// Fixed for the BladeRF.
const FFT_SIZE: usize = 8192;

// BLE channel width.
const CHANNEL_SIZE: u32 = 2_000_000;

const NUM_CHANNELS: usize = (SAMPLE_RATE / CHANNEL_SIZE) as usize;
const BIN_WIDTH: u32 = SAMPLE_RATE / FFT_SIZE as u32;
const BINS_PER_CHAN: usize = (CHANNEL_SIZE as f32 / BIN_WIDTH as f32).ceil() as usize;
const NUM_SAMPLES: usize = ((DURATION * SAMPLE_RATE as f32) / FFT_SIZE as f32).ceil() as usize;
const NUM_SAMPLES_WINDOW: usize = ((WINDOW * SAMPLE_RATE as f32) / FFT_SIZE as f32).ceil() as usize;

fn fft_shift<T>(data: &mut [T]) {
    data.rotate_left(FFT_SIZE / 2);
}

/// Builds desperado source based on Soapy abstraction layer.
async fn build_iq_source() -> IqAsyncSource {
    IqAsyncSource::from_soapy(
        "",
        SDR_CHANNEL,
        CENTER_FREQ,
        SAMPLE_RATE,
        Gain::Manual(GAIN),
        BANDWIDTH,
        "buffers=16,buflen=8192,transfers=8",
    )
    .await
    .unwrap()
}

/// Aggregate min, max and average per BLE channel.
fn calc_channel_stats<W: Write>(
    batch: &[Complex<f32>],
    file_writer: &mut W,
    out: &mut [(f32, f32, f32); NUM_CHANNELS],
) {
    for ch in 0..NUM_CHANNELS {
        let start = ch * BINS_PER_CHAN;
        let end = std::cmp::min(start + BINS_PER_CHAN, FFT_SIZE);
        let (sum, min, max) = batch[start..end]
            .iter()
            .filter_map(|c| {
                let x = 10.0 * c.norm_sqr().log10(); // 10*log10(∣c∣²)
                file_writer.write_all(&(x as f64).to_le_bytes()).unwrap();
                x.is_finite().then_some(x)
            })
            .fold(out[ch], |(sum, min, max), curr| {
                (sum + curr, f32::min(min, curr), f32::max(max, curr))
            });
        out[ch] = (sum, min, max);
    }
}

/// Processes a stream of data batches.
///
/// Note: the batches should already be in frequency-domain, i.e. parsed through an FFT.
///
/// It calculates the max, min, and average per channel per window size.
fn process_frequency_batches<'a, F: FnMut() -> T, T: AsRef<[Complex<f32>]>>(mut next: F) {
    let mut channel_writer = BufWriter::new(File::create("../data/channel_capture.dat").unwrap());

    #[cfg(feature = "save-capture")]
    let mut capture_writer = BufWriter::new(File::create("../data/capture.dat").unwrap());

    #[cfg(not(feature = "save-capture"))]
    let mut capture_writer = std::io::empty();

    let one_min = (60 * SAMPLE_RATE as usize / FFT_SIZE) / NUM_SAMPLES_WINDOW;

    for i in 0..(NUM_SAMPLES / NUM_SAMPLES_WINDOW) {
        if i % one_min == 0 {
            log::debug!("{}m", i / one_min)
        }

        let mut buffer = [(0f32, 0f32, 0f32); NUM_CHANNELS];
        for _ in 0..NUM_SAMPLES_WINDOW {
            let next = next();
            let next = next.as_ref();
            calc_channel_stats(next, &mut capture_writer, &mut buffer);
        }
        for (sum, min, max) in buffer {
            let avg = sum / (NUM_SAMPLES_WINDOW * BINS_PER_CHAN) as f32;
            channel_writer
                .write_all(&(avg as f32).to_le_bytes())
                .unwrap();
            channel_writer
                .write_all(&(min as f32).to_le_bytes())
                .unwrap();
            channel_writer
                .write_all(&(max as f32).to_le_bytes())
                .unwrap();
        }
    }
}

/// Reads samples in batches from the SDR and processes them immediately.
#[cfg(not(feature = "post-process"))]
fn read_parse_real_time(rt: Runtime) {
    let mut reader = rt.block_on(build_iq_source());
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let now = std::time::SystemTime::now();

    let read_and_fft = || {
        let mut next = rt.block_on(reader.next()).unwrap().unwrap();
        // Translate to frequency domain.
        fft.process(&mut next);
        fft_shift(&mut next);
        next
    };
    process_frequency_batches(read_and_fft);

    log::debug!(
        "Systime diff for reading+parsing: {:?}ms",
        now.elapsed().unwrap().as_millis()
    );
}

fn main() {
    env_logger::init();
    let rt = Runtime::new().unwrap();

    #[cfg(not(feature = "post-process"))]
    read_parse_real_time(rt);

    #[cfg(feature = "post-process")]
    post_process::read_parse(rt);
}
