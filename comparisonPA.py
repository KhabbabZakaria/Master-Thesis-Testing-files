#here we compare the Price Advantages of DRL and VWAP keeping TWAP as benchmark


import numpy as np
import matplotlib.pyplot as plt

def plot_stacked_bar(data, series_labels, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, title=None,
                     colors=None, grid=True, reverse=False):


    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    '''for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i], color=color))
        cum_size += row_data'''

    final = []

    for i in range(len(vwap)):
        final.append(model[i] - vwap[i])


    mydict = {final[0]:category_labels[0], final[1]:category_labels[1]}

    final2 = sorted(final, reverse=True)

    category_labels2 = []
    for i in range(len(final2)):
        category_labels2.append(mydict[final2[i]])
    


    plt.bar(ind, final2)

    if category_labels:
        color = colors[0] if colors is not None else None
        plt.xticks(ind, category_labels2, 
                            label=series_labels[0], color=color)

    plt.yticks([ 0, 0.25, .5, 0.75], ['0', '0.25' ,'0.5', '0.75'])

    if y_label:
        plt.ylabel(y_label)

    if title:
        plt.title(title)

    plt.legend()

    print(data)

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")





plt.figure(figsize=(6, 4))
plt.ylim(0, .9)


series_labels = ['VWAP', 'DRL']


vwap = [20.3, 1.47]

model = [20.8, 2.27]



data = [vwap, model]

category_labels =  ['NDAQ', 'DOW']


plot_stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=False, 
    value_format="{:.1f}",
    colors=['tab:orange', 'tab:green'],
    y_label="Performance improvements over VWAP",
    title="Composite Stocks During Normal Market"
)

#plt.savefig('bar.png')
plt.show()


