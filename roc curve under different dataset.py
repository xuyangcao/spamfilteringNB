import pylab

fpr_lemm_stop = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3125, 0.5208333333333333, 0.625, 0.75, 0.8125, 0.8333333333333334, 0.9166666666666666, 0.9166666666666666, 0.9583333333333334, 0.9583333333333334, 0.9791666666666666, 0.9791666666666666, 1]
tpr_lemm_stop = [0, 0.0, 0.004149377593360996, 0.012448132780082987, 0.029045643153526972, 0.058091286307053944, 0.1037344398340249, 0.2157676348547718, 0.43568464730290457, 0.6473029045643154, 0.8464730290456431, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1]

fpr_idf = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20833333333333337, 0.20833333333333337, 0.20833333333333337, 0.20833333333333337, 0.41666666666666663, 0.47916666666666663, 0.47916666666666663, 0.47916666666666663, 0.5208333333333333, 0.625, 0.625, 0.625, 0.6458333333333333, 0.6875, 0.7083333333333333, 0.7083333333333333, 0.7083333333333333, 0.7083333333333333, 0.8541666666666666, 0.8541666666666666, 0.8541666666666666, 0.8541666666666666, 0.875, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.8958333333333334, 0.9166666666666666, 0.9166666666666666, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 1.0, 1.0, 1.0, 1]
tpr_idf = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004149377593360996, 0.008298755186721992, 0.008298755186721992, 0.008298755186721992, 0.008298755186721992, 0.012448132780082987, 0.016597510373443983, 0.016597510373443983, 0.016597510373443983, 0.024896265560165973, 0.024896265560165973, 0.024896265560165973, 0.024896265560165973, 0.04564315352697095, 0.058091286307053944, 0.058091286307053944, 0.058091286307053944, 0.07053941908713693, 0.08298755186721991, 0.0995850622406639, 0.0995850622406639, 0.0995850622406639, 0.17427385892116182, 0.1991701244813278, 0.2033195020746888, 0.2033195020746888, 0.34439834024896265, 0.3983402489626556, 0.4066390041493776, 0.4066390041493776, 0.5228215767634855, 0.5601659751037344, 0.6597510373443983, 0.6597510373443983, 0.6597510373443983, 0.6639004149377593, 0.995850622406639, 0.995850622406639, 0.995850622406639, 0.995850622406639, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1]

fpr_bare = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10416666666666663, 0.39583333333333337, 0.5, 0.625, 0.7083333333333333, 0.7708333333333334, 0.8125, 0.875, 0.8958333333333334, 0.9166666666666666, 0.9583333333333334, 0.9583333333333334, 1]
tpr_bare = [0, 0.02074688796680498, 0.03734439834024896, 0.07053941908713693, 0.0954356846473029, 0.16597510373443983, 0.2821576763485477, 0.38589211618257263, 0.5103734439834025, 0.7178423236514523, 0.9087136929460581, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1]

fpr_stop = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.35416666666666663, 0.5208333333333333, 0.6041666666666667, 0.6458333333333333, 0.7291666666666667, 0.7916666666666666, 0.8333333333333334, 0.8333333333333334, 0.8333333333333334, 0.8541666666666666, 0.8541666666666666, 1]
tpr_stop =  [0, 0.008298755186721992, 0.016597510373443983, 0.07053941908713693, 0.12448132780082988, 0.18672199170124482, 0.3112033195020747, 0.4066390041493776, 0.5477178423236515, 0.6556016597510373, 0.8464730290456431, 0.991701244813278, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1]


pylab.plot(fpr_lemm_stop, tpr_lemm_stop, '.-', label = 'lemm_stop')
pylab.plot(fpr_idf, tpr_idf, '.-', label = 'lemm')
pylab.plot(fpr_bare, tpr_bare, '.-', label = 'bare')
pylab.plot(fpr_stop, tpr_stop, 'm.-', label = 'stop')

pylab.xlim(-0.1, 1.1)
pylab.ylim(-0.1, 1.1)           
pylab.legend(loc = 'best')
pylab.title('ROC curve under different dataset')
pylab.xlabel('FPR')
pylab.ylabel('TPR')

pylab.show()