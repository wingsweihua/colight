import runexp
import testexp
import summary


memo = "multi_phase/sumo/pipeline"
runexp.main(memo)
print("****************************** runexp ends (generate, train, test)!! ******************************")
summary.main(memo)
print("****************************** summary_detail ends ******************************")