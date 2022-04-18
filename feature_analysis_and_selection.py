import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import os
from matplotlib.pyplot import figure

OUTPUT_PATH = './output'
figure(figsize=(20, 20), dpi=80)

def check_existing_folder(this_path):
    MYDIR = (this_path)
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)

def plot_pca(ds, w, idx_thr):
    """

    :param ds: DataFrame with  features you have computed PCA
    :param w: weights of PCA
    :param idx_thr: n of PCA components to be plotted
    :return:
    """
    # plot weights of first x PCA components
    idx_thr_plot = 5
    barWidth = 0.15
    br0 = np.arange(len(w[0]))

    plt.figure()
    for i in range(idx_thr_plot):
        this_br = [x + i*barWidth for x in br0]
        plt.bar(this_br, w[i]/np.sqrt(sum(w[i]**2)), width=barWidth, label=f'PCA_{i+1}')

    plt.xticks([r+idx_thr_plot/2 * barWidth for r in br0],
               ds.columns.values, rotation=90)
    plt.legend(loc='best')
    plt.xlabel('Features')
    plt.ylabel('PCA weights')
    plt.title(f'First {idx_thr} PCA weights')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(OUTPUT_PATH, 'PCA_weights.eps'), bbox_inches="tight")



def check_perf(model, y, X, included, new_column):
    yp = model.predict(sm.add_constant(pd.DataFrame(X[included+[new_column]])))
    resid = y-yp
    rss = np.sum(resid**2)
    MSE = rss/(len(y)-2)
    return MSE


def stepwise_selection(X, y,
                       initial_list=[],
                       test_size=0.2,
                       test_sample='os',
                       verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        test_size - % of train set
        test_sample - ['is', 'os']
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    Logit ln(p/1-p) = beta_0 + beta_1 x_1 + ...
    """
    included = list(initial_list)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    score = []
    score_is = []
    score_os = []
    idx_chosen = -1
    i = 1
    while True:
        print(f'Number of features: {i}')
        i += 1
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_score_is = pd.Series(index=excluded)
        new_score_os = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y_train, sm.add_constant(pd.DataFrame(X_train[included+[new_column]]))).fit()
            # check performance is
            MSE_is = check_perf(model, y_train, X_train, included, new_column)
            #check performance os
            MSE_os = check_perf(model, y_test, X_test, included, new_column)
            new_score_is[new_column] = MSE_is
            new_score_os[new_column] = MSE_os
        if test_sample == 'is':
            use_this_score = new_score_is
        else:
            use_this_score = new_score_os
        best_feature = use_this_score.idxmin()
        best_MSE = use_this_score.min()
        included.append(best_feature)
        score_is.append(new_score_is[best_feature])
        score_os.append(new_score_os[best_feature])
        score.append(best_MSE)
        if verbose:
            print('Add  {:20} with MSE {:.6}'.format(best_feature, best_MSE))
        if (((best_MSE > min(score)) and (len(excluded)!=1)) and idx_chosen == -1):
            if verbose:
                print('Reached local minimum: current MSE {:.6}, previous MSE {:.6}'.format(best_MSE, score[-2]))
            idx_chosen = len(score) - 1
        if len(excluded) == 1:
            break

    plt.figure()
    plt.plot(range(1, 1 + len(score_is)), score_is, label='In-sample MSE')
    plt.plot(range(1, 1 + len(score_os)), score_os, label='Out-of-sample MSE')
    plt.axvline(idx_chosen,  color='k', linestyle='--', label='Selected')
    plt.legend()
    plt.xlabel('Number of features')
    plt.ylabel('MSE')
    plt.xticks(np.arange(1, 1 + len(score_is), 5))
    plt.savefig(os.path.join(OUTPUT_PATH, 'forward_stepwise_selection.eps'), bbox_inches="tight")
    return included[:idx_chosen]


def main():
    input_path = 'output'
    fpath = 'scaled_dataset.csv'
    check_existing_folder(OUTPUT_PATH)
    test_size = 0.2
    N_DAY_WINDOW = 3
    data = pd.read_csv(os.path.join(input_path, fpath))

    def skipna_std(df):
        return df.std(skipna=True)/np.sqrt(len(df)-sum(df.isna()))
    def skipna_mean(df):
        return df.mean(skipna=True)/np.sqrt(len(df)-sum(df.isna()))

    # check volatility per id
    # std_idx = data.groupby(by=['id'])['mood'].agg(skipna_std)
    # plt.plot(range(len(std_idx)), std_idx.values)
    # plt.xlabel('Patient')
    # correlation = data.corr()

    # 1. select_is
    data = data.loc[data.rand > test_size] # this is is data from is/os split for normalization
    mood_cols = [f'mood_{1+x}_day_before' for x in range(N_DAY_WINDOW)]
    mood_cols_scaled = [f'mood_{1+x}_day_before_scaled' for x in range(N_DAY_WINDOW)]


    # 3. stepwise regression
    y = data['mood'].values
    cols_2_drop = ['id', 'mood', 'mood_scaled', 'mood_before_mean', 'rand'] + mood_cols_scaled
    X = data.drop(columns=cols_2_drop)
    result = stepwise_selection(X, y, test_size=0.33)
    result = sorted(result)
    print('resulting features:')
    print(result)
    df_attributes = pd.DataFrame({'selected_attributes': result})
    # df_attributes.to_csv(os.path.join(OUTPUT_PATH, 'selected_attributes.csv'), index=False)

    #############################
    #PCA
    ds = data.drop(columns=cols_2_drop).copy()
    pca_features = ds.columns
    mat = ds.cov()
    l, w = np.linalg.eig(mat.values)
    #select eig that explain
    weights = l.cumsum()/l.sum()
    plt.plot(range(1, len(l)+1), weights)
    THR = 0.9
    idx_thr = np.where(weights<=THR)[0][-1]

    plot_pca(ds, w, idx_thr)

    feat_list = []
    for i in range(idx_thr):
        idx = np.where(w[i]>0.3)
        feat_list = feat_list + pca_features[idx].to_list()
    feat_list = list(set(feat_list))
    not_in_feat_selected = [idx for idx in feat_list if idx not in result]
    df_attributes = pd.concat([df_attributes, pd.DataFrame({'PCA': not_in_feat_selected})], ignore_index=True, axis=1)
    df_attributes.to_csv(os.path.join(OUTPUT_PATH, 'selected_attributes.csv'), index=False)


    # 5. plot correlation - full attributes
    # sort columns
    sorted_cols = ['mood'] + result + [idx for idx in data.columns if idx not in ['mood'] + result + ['id']]

    corr = data.loc[:, sorted_cols].corr()
    swarm_plot = sns.heatmap(corr,
                             xticklabels=corr.columns,
                             yticklabels=corr.columns)
    fig = swarm_plot.get_figure()
    fig.savefig(os.path.join(OUTPUT_PATH, 'correlation_full.eps'), bbox_inches="tight")
    plt.close(fig)

    # 6. plot correlation of selected features
    selected = ['mood'] + result
    corr = data.loc[:, selected].corr()
    this_plot = sns.heatmap(corr,
                       xticklabels=corr.columns,
                       yticklabels=corr.columns)
    fig = this_plot.get_figure()
    fig.savefig(os.path.join(OUTPUT_PATH, 'correlation_restr.eps'), bbox_inches="tight")
    plt.close(fig)



if __name__ == '__main__':
    main()