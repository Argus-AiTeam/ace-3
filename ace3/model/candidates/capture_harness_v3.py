"""Exact monitor-only derivative of the authenticated transaction harness."""

HARNESS = "ace3_decoder_layer0_token_engine_main.cpp"
CAPTURE_FLAG = "--capture-only"


def capture_source(original):
    text = original.decode("ascii")
    edits = [
        ("static bool transaction_mode;",
         "static bool transaction_mode;\nstatic bool capture_only = false;"),
        ('            else if(argument=="--state-out" && i+1<argc) state_out=argv[++i];',
         '            else if(argument=="--state-out" && i+1<argc) state_out=argv[++i];\n'
         '            else if(argument=="--capture-only") capture_only=true;'),
        ('        if(dir.empty() || raw_dir.empty())',
         '        if(capture_only && !transaction_mode)\n'
         '            throw std::runtime_error("capture-only requires transactional mode");\n'
         '        if(dir.empty() || raw_dir.empty())'),
        ('expected_trace[trace_count].value!=top.trace_f16_o)',
         '(!capture_only && expected_trace[trace_count].value!=top.trace_f16_o))'),
        ('expected_final[final_count].value!=top.final_f16_o)',
         '(!capture_only && expected_final[final_count].value!=top.final_f16_o))'),
        ('std::cout<<"DECODER_LAYER_TOKEN_TRANSACTION_PASS layer="',
         'std::cout<<(capture_only ? "DECODER_LAYER_TOKEN_TRANSACTION_CAPTURED layer=" :\n'
         '                         "DECODER_LAYER_TOKEN_TRANSACTION_PASS layer=")'),
    ]
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError("unsupported capture harness source: " + old)
        text = text.replace(old, new, 1)
    return text.encode("ascii")


def compatible_sources(actual, accepted, read):
    """Allow only this exact harness derivative; all arithmetic stays byte-identical."""
    if set(actual) != set(accepted):
        raise ValueError("runtime source closure mismatch")
    capture = False
    for name, original in accepted.items():
        if actual[name]["sha256"] == original["sha256"]:
            continue
        if name != HARNESS or read(actual[name]) != capture_source(read(original)):
            raise ValueError("runtime source closure mismatch: " + name)
        capture = True
    return capture
