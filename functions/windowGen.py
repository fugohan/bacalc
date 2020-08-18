import numpy as np

def windowGen(data, n=2,lookback=1):
    windows =[]
    y = []
    if lookback > n:
        print("lookback kann nicht größer sein als n!")
        
    # data is data var 
    for i in range(len(data)-(n)):
        frame = []
        for j in range(n):
            frame.append(data[i+j])
        windows.append( np.array(frame) ) 
        if lookback is 1:
            y.append(data[i+j+lookback])
        else:
            lookback_frame = []
            for k in range(lookback):
                lookback_frame.append(data[i+j+1-k])
                if k is (lookback-1):
                    lookback_frame.reverse()
                    y.append(lookback_frame)
                    lookback_frame = []
    return np.array(windows), np.array(y)