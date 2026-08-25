# Official Model24 systematic continuation quality report

Reviewed admissible baseline: `showcasecontinuations15c`. The unreviewed `486e5d848245` ancestry is explicitly excluded from acceptance and claim-bearing evidence.

The adjacent `batch.jsonl` is authoritative and preserves all 32 ordered cases, prompt serializations and token IDs, raw decoded outputs, stop reasons, every ACE-vs-PyTorch step comparison, FP16 KV lineage, diagnostic wall times, and failures.

Status: **completed_with_mismatches**; completed 32/32 cases and 122 generated steps, with 25 preserved comparison mismatches.

| # | Case | Category | Lang | Token IDs | Raw decoded output | Stop | Result |
|---:|---|---|---|---|---|---|---|
| 1 | `en_continuation_water_freezes` | continuation | en | `[279, 7329, 315, 279]` | `" the surface of the"` | max_new_tokens | pass |
| 2 | `en_continuation_opposite_hot` | continuation | en | `[1304, 12148, 32, 13]` | `" ____\nA."` | max_new_tokens | completed_with_mismatches |
| 3 | `en_continuation_triangle` | continuation | en | `[264, 46342, 315, 220]` | `" a perimeter of "` | max_new_tokens | completed_with_mismatches |
| 4 | `zh_continuation_year_months` | continuation | zh | `[109080, 3837, 109080, 18830]` | `"四季，四季有"` | max_new_tokens | pass |
| 5 | `zh_continuation_sun_rises` | continuation | zh | `[67364, 99319, 113635, 3837]` | `"东边升起，"` | max_new_tokens | completed_with_mismatches |
| 6 | `zh_continuation_one_plus_one` | continuation | zh | `[99195, 11319, 481, 10236]` | `"几？ - �"` | max_new_tokens | completed_with_mismatches |
| 7 | `en_chat_greeting` | chat | en | `[9707, 11, 1246, 1231]` | `"Hello, how may"` | max_new_tokens | pass |
| 8 | `en_chat_blue` | chat | en | `[10331, 374, 3545, 5815]` | `"Blue is often associated"` | max_new_tokens | pass |
| 9 | `en_chat_thanks` | chat | en | `[13060, 498, 0, 151645]` | `"Thank you!"` | eos_token | completed_with_mismatches |
| 10 | `zh_chat_greeting` | chat | zh | `[108386, 6313, 151645]` | `"你好！"` | eos_token | completed_with_mismatches |
| 11 | `zh_chat_red_object` | chat | zh | `[101053, 102716, 104165, 104613]` | `"一种常见的红色物品"` | max_new_tokens | pass |
| 12 | `zh_chat_good_night` | chat | zh | `[99438, 50285, 1773, 151645]` | `"晚安。"` | eos_token | completed_with_mismatches |
| 13 | `en_factual_largest_ocean` | factual | en | `[16462, 21575, 11, 892]` | `" Pacific Ocean, which"` | max_new_tokens | completed_with_mismatches |
| 14 | `en_factual_red_planet` | factual | en | `[279, 825, 448, 279]` | `" the one with the"` | max_new_tokens | completed_with_mismatches |
| 15 | `en_factual_week_days` | factual | en | `[2003, 11, 323, 1817]` | `" week, and each"` | max_new_tokens | completed_with_mismatches |
| 16 | `zh_factual_china_capital` | factual | zh | `[2130, 8997, 32, 13]` | `"____。\nA."` | max_new_tokens | completed_with_mismatches |
| 17 | `zh_factual_water_formula` | factual | zh | `[2130, 8997, 32, 13]` | `"____。\nA."` | max_new_tokens | pass |
| 18 | `zh_factual_four_seasons` | factual | zh | `[105419, 3837, 90919, 106084]` | `"季节，其中夏季"` | max_new_tokens | pass |
| 19 | `en_commonsense_rain` | commonsense | en | `[2679, 432, 374, 83253]` | `"If it is raining"` | max_new_tokens | pass |
| 20 | `en_commonsense_dark_room` | commonsense | en | `[641, 264, 6319, 3054]` | `"In a dark room"` | max_new_tokens | completed_with_mismatches |
| 21 | `en_commonsense_thirsty` | commonsense | en | `[32, 97108, 1697, 1265]` | `"A thirsty person should"` | max_new_tokens | pass |
| 22 | `zh_commonsense_cold` | commonsense | zh | `[104307, 109646, 13343, 3837]` | `"天气寒冷时，"` | max_new_tokens | pass |
| 23 | `zh_commonsense_cross_road` | commonsense | zh | `[38182, 109637, 24562, 99730]` | `"过马路前应该"` | max_new_tokens | pass |
| 24 | `en_code_python_square` | code | en | `[856, 353, 856, 271]` | `" x * x\n\n"` | max_new_tokens | pass |
| 25 | `en_code_javascript_add` | code | en | `[264, 488, 293, 280]` | `" a + b;\n"` | max_new_tokens | pass |
| 26 | `en_code_sql_select` | code | en | `[829, 20529, 7677, 82]` | `" name LIKE '%s"` | max_new_tokens | pass |
| 27 | `zh_code_python_sum` | code | zh | `[264, 488, 293, 271]` | `" a + b\n\n"` | max_new_tokens | completed_with_mismatches |
| 28 | `zh_code_python_even` | code | zh | `[308, 1018, 220, 17]` | `" n % 2"` | max_new_tokens | pass |
| 29 | `en_reasoning_apples` | reasoning | en | `[18, 10, 17, 28]` | `"3+2="` | max_new_tokens | pass |
| 30 | `zh_reasoning_subtract` | reasoning | zh | `[18, 151645]` | `"3"` | eos_token | pass |
| 31 | `zh_reasoning_sequence` | reasoning | zh | `[16, 15, 151645]` | `"10"` | eos_token | pass |
| 32 | `zh_reasoning_boxes` | reasoning | zh | `[24, 151645]` | `"9"` | eos_token | pass |

Scope: bounded software/oracle evidence only. Diagnostic host wall time is not product latency or throughput. No full-model RTL, FPGA, synthesis, PPA, broad dialogue-quality, latency, or throughput claim is made.
