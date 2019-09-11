import run_baseline
import summary

memo = "multi_phase/sumo/fixedtime_no_change_lane_optimal"
run_baseline.main(memo)
print("********************** run_baseline ends **********************")
summary.summary_detail_baseline(memo)
print("********************** run_baseline ends **********************")