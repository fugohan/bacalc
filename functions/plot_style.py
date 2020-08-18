import matplotlib.pyplot as plt
def plot_style(darkmode):
  if darkmode is True: 
    plt.style.use('dark_background')
  elif darkmode is False: 
    plt.style.use('default')