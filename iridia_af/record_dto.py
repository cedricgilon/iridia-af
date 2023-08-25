from dataclasses import dataclass

@dataclass
class RecordMetadata:
    hospital_id: str
    patient_id: str
    patient_sex: str
    patient_age: int
    record_id: str
    record_date: str
    record_start_time: str
    record_end_time: str
    record_timedelta: str
    record_n_files: int
    record_n_seconds: int
    record_n_samples: int
    type: str


@dataclass
class ECGEvent:
    # start event
    start_datetime: str
    start_file_index: int
    start_qrs_index: int
    # end event
    end_datetime: str
    end_file_index: int
    end_qrs_index: int
    # duration
    af_duration: int
    nsr_duration: int


@dataclass
class RREvent:
    # start event
    start_file_index: int
    start_rr_index: int
    # end event
    end_file_index: int
    end_rr_index: int