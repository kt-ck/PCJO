import numpy as np
def WilcoxonAUC(Sout,Nout):
    data=np.append(Sout.flatten('F'), Nout.flatten('F')).tolist()
    xs = list(set(data))
    xs.sort(reverse=True)
    
    tpf = np.zeros((len(xs)+1, 1))
    fpf = np.zeros((len(xs)+1, 1))

    cnt = 0
    for thresh in xs:
        tpf[cnt] = (np.sum(np.where(Sout > thresh, 1, 0))/ Sout.shape[0])
        fpf[cnt] = (np.sum(np.where(Nout > thresh, 1, 0))/ Nout.shape[0])
        cnt = cnt+1

    tpf[cnt] = 1
    fpf[cnt] = 1

    AUC = np.trapz(tpf.flatten('F'),fpf.flatten('F'))
    return AUC,tpf,fpf