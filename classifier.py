def calc_percentage(value, ref_value):
    return abs(value - ref_value) * 100 / ref_value


def bpm(bpm_value):
    result = {
        "normal": True,
        "tachycardia": [False, -1],
        "bradycardia": [False, -1]
              }
    if bpm_value > 100:
        result["normal"] = False
        result["tachycardia"][0] = True
        result["tachycardia"][1] = calc_percentage(bpm_value, 100)
    elif bpm_value < 60:
        result["normal"] = False
        result["bradycardia"][0] = True
        result["bradycardia"][1] = calc_percentage(bpm_value, 60)
    return result


def r_amp(r_amp_value):
    result = {
        "normal": True,
        "reduced_value": [False, -1],
        "increased_value": [False, -1]
    }
    if r_amp_value < 0.75:
        result["normal"] = False
        result["reduced_value"][0] = True
        result["reduced_value"][1] = calc_percentage(r_amp_value, 0.75)
    elif r_amp_value > 2:
        result["normal"] = False
        result["increased_value"][0] = True
        result["increased_value"][1] = calc_percentage(r_amp_value, 2)
    return result


def rr_interval(rr_value):
    result = {
        "normal": True,
        "shortened": [False, -1],
        "prolonged": [False, -1]
    }
    if rr_value > 1000:
        result["normal"] = False
        result["prolonged"][0] = True
        result["prolonged"][1] = calc_percentage(rr_value, 1000)
    elif rr_value < 600:
        result["normal"] = False
        result["shortened"][0] = True
        result["shortened"][1] = calc_percentage(rr_value, 600)
    return result


def hrv(hrv_parameters):
    sdnn = hrv_parameters[0]
    rmssd = hrv_parameters[1]
    result = {
        "normal": True,
        "sdnn_low": [False, -1],
        "rmssd_high": [False, -1],
        "rmssd_low": [False, -1]
    }
    if sdnn < 50:
        result["normal"] = False
        result["sdnn_low"][0] = True
        result["sdnn_low"][1] = calc_percentage(sdnn, 30)
    if rmssd < 20:
        result["normal"] = False
        result["rmssd_low"][0] = True
        result["rmssd_low"][1] = calc_percentage(rmssd, 20)
    elif rmssd > 90:
        result["normal"] = False
        result["rmssd_high"][0] = True
        result["rmssd_high"][1] = calc_percentage(rmssd, 90)
    return result


def qtc(qtc_value):
    result = {
        "normal": True,
        "shortened": [False, -1],
        "prolonged": [False, -1]
    }
    if qtc_value > 0.45:
        result["normal"] = False
        result["prolonged"][0] = True
        result["prolonged"][1] = calc_percentage(qtc_value, 450)
    elif qtc_value < 0.35:
        result["normal"] = False
        result["shortened"][0] = True
        result["shortened"][1] = calc_percentage(qtc_value, 350)
    return result


def st(st_parameters):
    st_interval = st_parameters[0]
    st_coeff = st_parameters[1]
    j_amp = st_parameters[2]
    st_state = st_parameters[3]
    result = {
        "normal": True,
        "abnormal_j_amp": [False, -1],
        "shortened_st_interval": [False, -1],
        "prolonged_st_interval": [False, -1],
        "positive_curvature": False,
        "elevated": False,
        "depressed": False
    }
    if st_interval < 80:
        result["normal"] = False
        result["shortened_st_interval"][0] = True
        result["shortened_st_interval"][1] = calc_percentage(st_interval, 80)
    elif st_interval > 120:
        result["normal"] = False
        result["prolonged_st_interval"][0] = True
        result["prolonged_st_interval"][1] = calc_percentage(st_interval, 120)
    if st_coeff > 0:
        result["positive_curvature"] = True
    if abs(j_amp) > 0.1:
        result["normal"] = False
        result["abnormal_j_amp"][0] = True
        result["abnormal_j_amp"][1] = calc_percentage(abs(j_amp), 0.5)
    if st_state > 0.1 or j_amp > 0.05:
        result["normal"] = False
        result["elevated"] = True
    elif st_state < -0.1 or j_amp < -0.05:
        result["normal"] = False
        result["depressed"] = True
    return result


def qrs_duration(duration):
    result = {
        "normal": True,
        "shortened": [False, -1],
        "prolonged": [False, -1]
    }
    if duration > 120:
        result["normal"] = False
        result["prolonged"][0] = True
        result["prolonged"][1] = calc_percentage(duration, 120)
    elif duration < 90:
        result["normal"] = False
        result["shortened"][0] = True
        result["shortened"][1] = calc_percentage(duration, 90)
    return result


def collect_and_test(features, st_features):
    bpm_card = bpm(features["bpm"])
    r_amp_card = r_amp(features["avr_r_amp"])
    rr_interval_card = rr_interval(features["avr_rr_interval"])
    hrv_list = [features["sdnn"], features["rmssd"]]
    hrv_card = hrv(hrv_list)
    qtc_card = qtc(features["qtc"])
    qrs_duration_card = qrs_duration(features["avr_qrs_duration"])
    st_list = [features["avr_st_interval"], 0, st_features[0]["avr_j_value"], st_features[0]["avr_st_value"]]
    st_card = st(st_list)
    cards = {
        "bpm_card": bpm_card,
        "r_amp_card": r_amp_card,
        "rr_interval_card": rr_interval_card,
        "hrv_card": hrv_card,
        "qtc_card": qtc_card,
        "qrs_duration_card": qrs_duration_card,
        "st_card": st_card
    }
    print(cards)
    arrhythmia_test = arrhythmia(bpm_card, hrv_card, qrs_duration_card)
    bundle_brunch_block_test = bundle_brunch_block(qtc_card, qrs_duration_card, bpm_card, r_amp_card)
    mi_test = mi(st_card, qtc_card,  bpm_card, r_amp_card)
    diagnosis = {
        "arrhythmia_test": arrhythmia_test,
        # "lhv_test": lhv_test,
        "bundle_brunch_block_test": bundle_brunch_block_test,
        "mi_test": mi_test
    }
    return diagnosis


def arrhythmia(bpm_value, hrv_values, qrs_value):
    score = 0
    flag = False
    if not bpm_value["normal"]:
        score += 1
    if hrv_values["sdnn_low"][0]:  # if no p wave (af) definite arrhythmia
        score += 1
    if hrv_values["rmssd_low"][0] or hrv_values["rmssd_high"][0]:  # if no p wave (af) definite arrhythmia
        score += 1
    if qrs_value["prolonged"][0]:
        score += 1
    if score >= 2:
        flag = True
    return flag, score


def bundle_brunch_block(qtc_value, qrs_value, bpm_value, r_value):
    score = 0
    flag = False
    if qrs_value["prolonged"][0]:
        score += 3
    if not bpm_value["normal"]:
        score += 1
    if r_value["reduced_value"][0]:
        score += 2
    if qtc_value["prolonged"][0]:
        score += 2
    if score >= 3:
        flag = True
    return flag, score


def mi(st_values, qtc_value, bpm_value, r_value):
    score = 0
    flag = False
    if st_values["elevated"] or st_values["depressed"]:
        score += 3
    if not bpm_value["normal"]:
        score += 1
    if r_value["reduced_value"][0]:
        score += 2
    if qtc_value["prolonged"][0]:
        score += 2
    if score >= 3:
        flag = True
    return flag, score
