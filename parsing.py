import re
from datetime import timezone, datetime


def parse_goes_filename(fname):
    """Parses attributes of file available in filename

    The main categories are separated by underscores:
    ops environment, Data Set Name (DSN), platform, start, end, created


    The DSN has multiple sub-fields, e.g.: DSN for the
    Lighting Detection product is “GLM-L2-LCFA”

    e.g.
    OR_ABI-L2-MCMIPF-M6_G16_s20201800021177_e20201800023556_c20201800024106.nc
    OR: Operational System Real-Time Data
    ABI-L2: Advanced Baseline Imager Level 2+ (other option is level 1, L1a, L1b)
    CMIPF: product. Cloud and Moisture Image Product – Full Disk
    M3/M4/M6: ABI Mode 3, 4 or 6, (Mode 6=10 minute flex)
    C09: Channel Number (Band 9 in this example)
    G16: GOES-16
    sYYYYJJJHHMMSSs: Observation Start
    eYYYYJJJHHMMSSs: Observation End
    cYYYYJJJHHMMSSs: File Creation

    L1 example:
    OR_ABI-L1b-RadC-M3C01_G16_s20190621802131_e20190621804504_c20190621804546.nc
    Lightning mapper:
    OR_GLM-L2-LCFA_G16_s20190720000000_e20190720000200_c20190720000226.nc

    Reference:
    https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
    Appendix A
    """
    fname_noslash = fname.split("/")[-1]

    fname_pattern = re.compile(
        r"OR_"
        r"(?P<DSN>.*)_"
        r"(?P<platform>G\d+)_"
        r"s(?P<start_time>\w+)_"
        r"e(?P<end_time>\w+)_"
        r"c(?P<creation_time>\w+).nc"
    )
    time_pattern = "%Y%j%H%M%S%f"
    m = re.match(fname_pattern, fname_noslash)
    if not m:
        raise ValueError(f"{fname_noslash} does not match GOES format")

    time_pattern = "%Y%j%H%M%S%f"
    match_dict = m.groupdict()
    for k in ("start_time", "end_time", "creation_time"):
        dt = datetime.strptime(match_dict[k], time_pattern)
        match_dict[k] = dt.replace(tzinfo=timezone.utc)

    if "ABI" in match_dict["DSN"]:
        dsn_pattern = re.compile(
            r"(?P<instrument>ABI|GLM)-"  # fix this
            r"(?P<level>L[12b]+)-"
            r"(?P<product>\w+)-"
            r"(?P<mode>M[3-6AM])"
            r"(?P<channel>C\d+)?"  # channel is optional
        )
        m2 = re.match(dsn_pattern, match_dict["DSN"])
        match_dict.update(m2.groupdict())
    return match_dict
