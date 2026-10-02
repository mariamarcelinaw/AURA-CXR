import statistics
def final_epoch(vals): return max(1,int(round(float(statistics.median(vals)))))
def test_xrv_eva_rules(): assert final_epoch([1,2,2,1,2])==2; assert final_epoch([2,1,1,2,3])==2
def test_tf_phase_rules(): assert final_epoch([40,37,30,1,39])==37; assert final_epoch([15,15,20,2,20])==15
