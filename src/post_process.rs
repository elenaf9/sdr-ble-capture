use std::mem::MaybeUninit;

use futures::stream::StreamExt;
use num_complex::Complex;
use rustfft::FftPlanner;
use tokio::runtime::Runtime;

use super::{FFT_SIZE, NUM_SAMPLES};


/// Read all samples from the SDR.
async fn read_samples(buffer: &mut [MaybeUninit<Vec<Complex<f32>>>; NUM_SAMPLES]) {
    let mut reader = super::build_iq_source().await;
    let now = std::time::SystemTime::now();
    for batch in buffer {
        let next = reader.next().await.unwrap().unwrap();
        batch.write(next);
    }
    log::debug!("reading {:?}ms", now.elapsed().unwrap().as_millis());
}

/// Processes all samples with an FFT and writes the results to the buffer.
fn fft_transform_samples(
    samples: &mut [Vec<Complex<f32>>; NUM_SAMPLES],
    buffer: &mut [[Complex<f32>; FFT_SIZE]; NUM_SAMPLES],
) {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    for (i, batch) in samples.iter_mut().enumerate() {
        fft.process(batch);
        super::fft_shift(batch);
        for (j, c) in batch.iter().enumerate() {
            buffer[i][j] = *c
        }
    }
}

/// Reads all samples from the SDR first, then processes the collected data.
pub fn read_parse(rt: Runtime) {
    let mut samples = Box::new(MaybeUninit::uninit());
    rt.block_on(read_samples(samples.as_mut().as_mut()));
    let mut samples = unsafe { samples.assume_init() };

    let now = std::time::SystemTime::now();
    let mut frequencies = Box::new([[const { Complex::ZERO }; FFT_SIZE]; NUM_SAMPLES]);
    fft_transform_samples(&mut samples, &mut frequencies);

    let mut frequencies_iter = frequencies.iter_mut();

    let next = || frequencies_iter.next().unwrap();
    super::process_frequency_batches(next);

    log::debug!("parsing {:?}ms", now.elapsed().unwrap().as_millis());
}
