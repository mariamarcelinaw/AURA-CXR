from aura_cxr.xai import assert_xai_config,XAI_SCOPE
def test_xai(): assert assert_xai_config(); assert XAI_SCOPE=='HYBRID_CNN_VIT_BRANCH_ONLY'
