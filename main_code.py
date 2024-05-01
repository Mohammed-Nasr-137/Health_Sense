import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ecgdetectors import Detectors
import os
import feature_extraction
import classifier

directory = ("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\ptb-diagnostic-ecg-database-1.0.0"
             "\Converted2\\Healthy\\")
fs = 1000
# data = pd.read_csv("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\MITBIH\\100.csv")
# data = pd.read_csv("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\ptb-diagnostic-ecg-database-1.0.0"
#                    "\Converted2\\mi\\s0075lre.csv")
detectors = Detectors(fs)


def plot_ecg(signal, title, arr=None, mark=False, label=None):
    plt.figure(figsize=(12, 5))
    plt.plot(signal)
    plt.xlabel('Time (samples)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)
    if mark:
        time_axis = np.arange(len(signal))
        plt.scatter(time_axis[arr], signal[arr], c="red", marker="x", label=label)
        plt.legend()
    plt.grid(True)
    plt.show()


def moving_average_filter(input_data, window_size):
    if window_size > len(input_data):
        raise ValueError("Window size cannot be larger than data length")
    filtered_data = np.zeros(len(input_data))
    for p in range(len(input_data)):
        start_index = max(0, p - window_size // 2)
        end_index = min(len(input_data), p + window_size // 2 + 1)
        window_data = input_data[start_index:end_index]
        filtered_data[p] = np.mean(window_data)
    return filtered_data


def two_stage_median_filter(ecg_data, freq):
    window_size_stage1 = np.linspace(freq / 4, freq / 2, len(ecg_data))  # Adjust window size as needed
    filtered_data_stage1 = np.zeros_like(ecg_data)
    for i in range(len(ecg_data)):
        window_start = max(0, i - int(window_size_stage1[i] / 2))
        window_end = min(len(ecg_data), i + int(window_size_stage1[i] / 2) + 1)
        window_data = ecg_data[window_start:window_end]
        filtered_data_stage1[i] = np.median(window_data)
    window_size_stage2 = freq
    filtered_data_stage2 = np.zeros_like(ecg_data)
    for i in range(len(ecg_data)):
        window_start = max(0, i - int(window_size_stage2 / 2))
        window_end = min(len(ecg_data), i + int(window_size_stage2 / 2) + 1)
        window_data = filtered_data_stage1[window_start:window_end]
        filtered_data_stage2[i] = np.median(window_data)
    baseline_removed_ecg = ecg_data - filtered_data_stage2
    return baseline_removed_ecg


def nlms_filter(inputdata, desired_signal, mu):
    n = len(inputdata)
    m = 5
    filtered_data = inputdata.copy()
    for n in range(n):
        start_index = max(0, n - m)
        end_index = min(n, n + m + 1)
        data_window = inputdata[start_index:end_index]
        error = desired_signal[n] - np.dot(data_window, filtered_data[start_index:end_index])
        if np.linalg.norm(data_window) > 0:
            normalized_step = mu / (np.linalg.norm(data_window) ** 2)
        else:
            normalized_step = 0
        filtered_data[n] = filtered_data[n] + normalized_step * error
    return filtered_data


def snr(noisy_signal, filtered):
    estimated_noise_power = np.mean((np.subtract(noisy_signal, filtered)) ** 2)
    squared = []
    for element in noisy_signal:
        squared.append(element ** 2)
    signal_power = np.mean(squared) - estimated_noise_power
    snr_db = 10 * np.log10(signal_power / estimated_noise_power)
    return snr_db


def peak_detection(ecg, fraction=0.5, r_window=240):
    max_peak = np.max(ecg)
    threshold = fraction * max_peak
    peak_location = []
    last_peak_index = 0
    for i in range(len(ecg)):
        if ecg[i] >= threshold and (i - last_peak_index) >= r_window:
            search_window_start = max(0, i - 100)
            search_window_end = min(len(ecg), i + 100)
            peak_idx = np.argmax(np.abs(ecg[search_window_start:search_window_end])) + search_window_start
            try:
                if ecg[peak_idx] > ecg[peak_idx - 1] and ecg[peak_idx] > ecg[peak_idx + 1]:
                    peak_location.append(peak_idx)
                    last_peak_index = peak_idx
            except IndexError:
                peak_location.append(peak_idx)
                last_peak_index = peak_idx
    return peak_location


def pantompkins_preprocessing(signal):
    differentiated_signal = np.diff(signal)
    squared_signal = np.square(differentiated_signal)
    output_signal = moving_average_filter(squared_signal, 30)
    return output_signal


def peak_detection_test(pan_peaks, reference_peaks):
    difference = np.subtract(reference_peaks, pan_peaks)
    plot_ecg(difference, "Error")


counter = 0
record = []
successes = 0
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        try:
            data = pd.read_csv(directory + filename)
            data = data.fillna(data.mean())
            raw_ecg_signal = data["v5"].to_numpy()
            corrected_raw_signal = two_stage_median_filter(raw_ecg_signal, fs)
            # plot_ecg(corrected_raw_signal, "corrected_raw_signal")
            filtered_signal = nlms_filter(corrected_raw_signal, corrected_raw_signal, 0.000000000000000000000000025)
            # plot_ecg(filtered_signal, "filtered_signal")
            pantompkins_ecg = pantompkins_preprocessing(filtered_signal)
            peaks = peak_detection(pantompkins_ecg)
            features = feature_extraction.qrs_features(peaks, filtered_signal)
            st = feature_extraction.st_segment_shape_analysis(filtered_signal, fs, features["j_points"][:-1],
                                                              features["t_peaks"])
            # plot_ecg(filtered_signal, "pan_peaks", peaks, True, "R peaks")
            # plot_ecg(filtered_signal, "q", features["q_peaks"], True, "Q waves")
            # plot_ecg(filtered_signal, "s", features["s_peaks"], True, "S waves")
            # plot_ecg(filtered_signal, "t", features["t_peaks"], True, "T waves")
            # plot_ecg(filtered_signal, "J", features["j_points"], True, "J")
            diagnosis = classifier.collect_and_test(features, st)
            print(counter + 1, filename,  ": ", diagnosis)
            ask = int(input("record this diagnose?: "))
            # ask = 1
            if ask == 1:
                counter += 1
                record.append((filename, diagnosis))
                if (not diagnosis["bundle_brunch_block_test"][0] and not diagnosis["arrhythmia_test"][0] and
                        not diagnosis["mi_test"][0]):
                    print("Healthy!")
                    successes += 1
        except RuntimeWarning or IndexError:
            continue

print("total number of tests: %i" % counter)
print("True positives: %i" % successes)
accuracy = successes / counter
print("accuracy: %f" % accuracy)
print(record)
# print("nlms:  " + str(snr(corrected_raw_signal, filtered_signal)))
# plot_ecg(pantompkins_ecg, "preprocessed_ecg")
# plot_ecg(pantompkins_ecg, "peaks_ecg", peaks, True)
# ref_peaks = detectors.engzee_detector(filtered_signal)
# print(features)
# print(st)
# plot_ecg(filtered_signal, "ref_peaks", ref_peaks, True, "R peaks")
