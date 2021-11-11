import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')

def graficas_resultados(X_,y_, model):
    predictions =  model.predict(X_)
    result = pd.DataFrame({'predict': predictions})
    result['y_test'] = y_.tolist()

    X_plot = y_.to_numpy()#np.linspace(0, 7, 100)
    Y_plot = y_.to_numpy()#10*X_plot+5

    g = sns.FacetGrid(result)
    g = g.map(plt.scatter, "y_test", "predict", edgecolor="w")
    plt.plot(X_plot, Y_plot, color='r')
    plt.show();
    

def launch_model(name,model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ypred_train = model.predict(X_train)
    print ('MSE train', metrics.mean_absolute_error(y_train, ypred_train))
    print ('MSE test', metrics.mean_absolute_error(y_test, y_pred))
    r_1 = model.score(X_train, y_train)
    r_2 = model.score(X_test, y_test)

    print ('R^2 train', r_1)
    print ('R^2 test', r_2)
    #print('Tiempo de ejecución: {0:.2f} segundos.'.format(time.time() - start))
    return name + '($R^2={:.3f}$)'.format(r_2), np.array(y_test), y_pred



def plot(results):
    '''
    Create a plot comparing multiple learners.
    `results` is a list of tuples containing:
        (title, expected values, actual values)
    
    All the elements in results will be plotted.
    '''

    # Using subplots to display the results on the same X axis
    fig, plts = plt.subplots(nrows=len(results), figsize=(8, 8))
    fig.canvas.set_window_title('Predicting car values')

    # Show each element in the plots returned from plt.subplots()
    for subplot, (title, y, y_pred) in zip(plts, results):
        # Configure each subplot to have no tick marks
        # (these are meaningless for the sample dataset)
        subplot.set_xticklabels(())
        subplot.set_yticklabels(())

        # Label the vertical axis
        subplot.set_ylabel('Precio auto')

        # Set the title for the subplot
        subplot.set_title(title)

        # Plot the actual data and the prediction
        subplot.plot(y, 'b', label='actual')
        subplot.plot(y_pred, 'r', label='predicted')
        
        # Shade the area between the predicted and the actual values
        subplot.fill_between(
            # Generate X values [0, 1, 2, ..., len(y)-2, len(y)-1]
            np.arange(0, len(y), 1),
            y,
            y_pred,
            color='r',
            alpha=0.2
        )

        # Mark the extent of the training data
        subplot.axvline(len(y) // 2, linestyle='--', color='0', alpha=0.2)

        # Include a legend in each subplot
        subplot.legend()

    # Let matplotlib handle the subplot layout
    fig.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

   

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()