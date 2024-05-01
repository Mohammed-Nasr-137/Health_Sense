import numpy as np

fs = 1000


def identify_s_wave(ecg_data):
    potential_s_waves = np.min(ecg_data)
    if potential_s_waves is None:
        return -1
    return np.where(ecg_data == potential_s_waves)[0][0]


def identify_q_wave(ecg_data):
    potential_q_waves = np.min(ecg_data)
    if potential_q_waves is None:
        return -1
    return np.where(ecg_data == potential_q_waves)[0][0]


def identify_j_wave(ecg_data, s_wave):
    window = ecg_data[s_wave:s_wave + 70]
    j_point = np.max(window)
    return np.where(ecg_data == j_point)[0][0]


def identify_t_wave(ecg_data):
    potential_t_waves = np.max(ecg_data)
    if potential_t_waves is None:
        return -1
    return np.where(ecg_data == potential_t_waves)[0][0]


def st_segment_shape_analysis(ecg_data, freq, j_peaks, t_peaks, margins=0.02):
    j_values = []
    for i in j_peaks:
        j_values.append(ecg_data[i])
    avr_j_value = np.mean(j_values)
    st_values = []
    for i in range(min(len(j_peaks), len(t_peaks)) - 1):
        window = ecg_data[j_peaks[i]:t_peaks[i] - 70]
        st_values.append(np.mean(window))
    avr_st_value = np.mean(st_values)
    results = []
    for j_peak, t_peak in zip(j_peaks, t_peaks):
        st_segment_start = int(j_peak + margins * freq)
        st_segment_end = int(t_peak - margins * freq)
        st_segment_data = ecg_data[st_segment_start:st_segment_end]
        x = np.arange(len(st_segment_data))
        coefficients = np.linalg.lstsq(np.vstack([x ** 2, x, np.ones(len(x))]).T, st_segment_data, rcond=None)[0]
        curvature = 2 * coefficients[0]
        results.append({"coefficients": coefficients.tolist(), "curvature": curvature, "avr_j_value": avr_j_value,
                        "avr_st_value": avr_st_value})
    return results


def qrs_features(peak_location, signal):
    bpm = int(60 * len(peak_location) / (len(signal) / 1000))
    r_amp = []
    rr_intervals = np.diff(peak_location)
    avr_RR = np.mean(rr_intervals)
    sdnn = np.sqrt(np.var(rr_intervals))
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    time_to_peak = []
    s_peaks = []
    q_peaks = []
    t_peaks = []
    j_points = []
    s_r_ratios = []
    qrs_durations = []
    st_interval = []
    qt_interval = []
    window_size = 70
    for peak in peak_location:
        window_start = max(0, peak - int(window_size))
        window_end = min(len(signal), peak + int(window_size))
        r_amp.append(signal[peak])
        s_index = identify_s_wave(signal[peak:window_end]) + peak
        s_peaks.append(s_index)
        q_index = identify_q_wave(signal[window_start:peak]) + window_start
        q_peaks.append(q_index)
        t_index = identify_t_wave(signal[s_index:s_index + 400]) + s_index
        t_peaks.append(t_index)
        j_point = s_index + 30
        j_points.append(j_point)
        st_interval.append(t_index - s_index)
        time_to_peak.append(peak - q_index)
        qrs_durations.append(s_index - q_index + 45)
        qt_interval.append(t_index - q_index)
        s_r_ratios.append(abs(signal[s_index] / signal[peak]))
    avr_r_amp = np.mean(r_amp)
    avr_time_to_peak = np.mean(time_to_peak)
    avr_s_r_ratios = np.mean(s_r_ratios)
    avr_qrs_duration = np.mean(qrs_durations)
    avr_st_interval = np.mean(st_interval)
    avr_qt_interval = np.mean(qt_interval)
    qtc = (avr_qt_interval / 1000) / ((avr_RR / 1000) ** 0.5)
    features = {
        "bpm": bpm,
        "avr_r_amp": avr_r_amp,
        "avr_rr_interval": avr_RR,
        "q_peaks": q_peaks,
        # "p_peaks": p_peaks,
        "s_peaks": s_peaks,
        "t_peaks": t_peaks,
        "j_points": j_points,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "avr_time_to_peak": avr_time_to_peak,
        "avr_s_r_ratios": avr_s_r_ratios,
        "avr_qrs_duration": avr_qrs_duration,
        "avr_st_interval": avr_st_interval,
        "qtc": qtc
    }
    return features
