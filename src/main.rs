use std::{
    fs::File,
    io::{BufWriter, Write},
    mem::MaybeUninit,
};

use desperado::{Gain, IqAsyncSource};
use futures::stream::StreamExt;
use num_complex::Complex;
use rustfft::FftPlanner;

const FFT_SIZE: usize = 4096;

// SDR config
const CHANNEL: usize = 0;
const CENTER_FREQ: u32 = 2_420_000_000;
const SAMPLE_RATE: u32 = 40_000_000;
const GAIN: f64 = 50.0;
const BANDWIDTH: u32 = SAMPLE_RATE;

const DURATION: f32 = 10.0;

const NUM_SAMPLES: usize = ((DURATION * SAMPLE_RATE as f32) / FFT_SIZE as f32) as usize;

fn fft_shift(data: &mut [Complex<f32>]) {
    data.rotate_left(FFT_SIZE / 2);
}

async fn read_samples(buffer: &mut [MaybeUninit<Vec<Complex<f32>>>; NUM_SAMPLES]) {
    let mut reader = IqAsyncSource::from_soapy(
        "",
        CHANNEL,
        CENTER_FREQ,
        SAMPLE_RATE,
        Gain::Manual(GAIN),
        BANDWIDTH,
    )
    .await
    .unwrap();
    for batch in buffer {
        let next = reader.next().await.unwrap().unwrap();
        batch.write(next);
    }
}

fn fft_transform(
    samples: &mut [Vec<Complex<f32>>; NUM_SAMPLES],
    buffer: &mut [[f32; FFT_SIZE]; NUM_SAMPLES],
) {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    for (i, batch) in samples.iter_mut().enumerate() {
        fft.process(batch);
        fft_shift(batch);

        for (j, c) in batch.iter().enumerate() {
            buffer[i][j] = 10.0 * c.norm_sqr().log10(); // 10*log10(∣c∣²)
        }
    }
}

#[allow(unused)]
fn reduce_samples(data: &[[f32; FFT_SIZE]], f: fn(f32, f32) -> f32) -> f32 {
    data.iter().flatten().fold(0.0, |acc, curr| f(acc, *curr))
}

#[tokio::main]
async fn main() {
    let mut samples = MaybeUninit::uninit();
    read_samples(samples.as_mut()).await;
    let mut samples = unsafe { samples.assume_init() };

    let mut frequencies = Box::new([[0.0; FFT_SIZE]; NUM_SAMPLES]);
    fft_transform(&mut samples, &mut frequencies);

    let mut writer = BufWriter::new(File::create("../data/capture.dat").unwrap());

    for sample in frequencies.as_slice().as_flattened() {
        writer.write_all(&(*sample as f64).to_le_bytes()).unwrap();
    }
}
