import unittest
import numpy as np
import labcore as l

class NumericalTests(unittest.TestCase):
    def test_activity_deterministic(self):
        a=l.activity_data();b=l.activity_data();np.testing.assert_equal(a['hours'],b['hours'])
    def test_regression_learns(self):
        loss=l.regression()[3];self.assertLess(loss[-1],loss[0])
    def test_confusion_accounts_for_all_rows(self):
        r=l.classification();self.assertEqual(sum(r[k] for k in ['tp','fp','fn','tn']),90)
    def test_threshold_recall_monotonic(self):
        self.assertGreaterEqual(l.classification(threshold=.2)['recall'],l.classification(threshold=.8)['recall'])
    def test_xor_trains(self):
        loss,p,*_=l.train_xor(1000);self.assertLess(loss[-1],.05);np.testing.assert_array_equal(p>.5,[0,1,1,0])
    def test_convolution_shape(self):
        self.assertEqual(l.conv2(np.ones((5,5)),np.ones((3,3))).shape,(3,3))
    def test_attention_normalised_and_causal(self):
        a,_=l.attention(6,causal=True);np.testing.assert_allclose(a.sum(1),1);self.assertEqual(np.triu(a,1).sum(),0)
    def test_autoencoder_learns(self):
        _,_,loss=l.autoencoder();self.assertLess(loss[-1],loss[0])
    def test_softmax(self):
        _,p,h=l.temperature_distribution();self.assertAlmostEqual(sum(p),1);self.assertGreater(h,0)
    def test_retrieval(self):
        self.assertEqual(l.retrieve('monitor latency drift',1)[0][0],'OPS')
    def test_abstention(self):
        text,_=l.answer('zzzzzzz',2,.1);self.assertIn('Not enough evidence',text)
    def test_review_gate(self):
        self.assertIn('WAIT',l.agent_trace(8,True,False)[-1]);self.assertEqual(l.agent_trace(8,True,True)[-1],'final')
    def test_budget(self):
        self.assertIn('STOP',l.agent_trace(1,False,False)[-1])
    def test_same_distribution_psi(self):
        x=np.arange(30);self.assertAlmostEqual(l.psi(x,x)[0],0)
    def test_payload(self):
        self.assertEqual(l.serving({'hours':4,'practice':6})['score'],48)
    def test_reject_boolean_and_nan(self):
        for v in [True,float('nan'),float('inf'),'4',-1,25]:
            with self.assertRaises(ValueError):l.serving({'hours':v,'practice':6})
    def test_rollout_gate(self):
        self.assertTrue(l.rollout());self.assertFalse(l.rollout(latency=300));self.assertFalse(l.rollout(new_error=.2))

from pipeline import evaluate
class ProjectTests(unittest.TestCase):
    def test_fixed_artifact(self):
        a,r=evaluate();b,s=evaluate();self.assertEqual(a,b);self.assertEqual(r,s);self.assertEqual(a['test_rows'],80);self.assertEqual(len(r['artifact_sha256']),64)
    def test_accounting(self):
        a,r=evaluate();self.assertEqual(sum(r['test'][k] for k in ['tp','fp','fn','tn']),80)

if __name__=="__main__":unittest.main()
