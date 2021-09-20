import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('MacOSX')
from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import math
from utils.column_names import *
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve,auc


def roc_plot(df:pd.DataFrame,path:Optional[str]=None,TRUE_COL:str='actuals',PROB_COL:str='probabilities',logging:bool=False,)->int:
    plt.title("ROC")
    fpr,tpr,\
    thresholds=roc_curve(df[TRUE_COL].values,df[PROB_COL].values)
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label='ROC_AUC=%0.2f'%roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    if logging:
        plt.savefig(path+"roc_curve.png")
    return roc_auc

def calc_lift_table(df:pd.DataFrame,actual_col:str,prob_col:str,n_bins:int=10)->pd.DataFrame:
    df.sort_values(by=[prob_col],ascending=False,inplace=True)
    rows=[]
    for group in np.array_split(df,n_bins):
        score=group[actual_col].sum()

        rows.append({"BIN_CALCS":len(group),"LIFT_CORRECT":score})
    lift=pd.DataFrame(rows)
    lift['RANDOM_COUNT']=(float)(lift['LIFT_CORRECT'].sum())/len(lift)
    lift['LIFT_CALC']=lift['LIFT_CORRECT']/lift['RANDOM_COUNT']
    lift['CUM_LIFT_CALC']=lift["LIFT_CORRECT"].cumsum()/lift["RANDOM_COUNT"].cumsum()
    return lift


def plot_lift_chart(df:pd.DataFrame,TRUE_COL:str='actuals',PROB_COL:str='probabilities',logging:bool=False,path:Optional[str]=None,n_bins:int=20)->int:
    transformed_lift=calc_lift_table(df,TRUE_COL,PROB_COL,n_bins=10)
    plt.plot(transformed_lift['LIFT_CALC'],label='Response rate with')
    plt.title("Lift Chart")
    plt.ylabel("Lift")
    plt.xlabel("Percentile")

    plt.annotate("",xy=(0,1),xycoords="data",xytext=(len(transformed_lift['LIFT_CALC'])-1,1),textcoords='data',arrowprops=dict(arrowstyle='-',ls='dashed',color='r'))
    plt.xticks(np.arange(len(transformed_lift['LIFT_CALC'])),
               np.arange(100/transformed_lift.shape[0],101,step=100/transformed_lift.shape[0]))
    if logging:
        plt.savefig(path+"lift_chart.png")
    return transformed_lift['LIFT_CALC']

def roundup(x):
    return 100+int(math.ceil(x/100.0))*100

def round_up(n,decimals=0):
    multiplier=10**decimals
    return math.ceil(n*multiplier)/multiplier

def display_figures(ax,aList):
    i=0
    for p in ax.patches:
        h=p.get_height()
        if(h>0):
            ax.text(p.get_x()+p.get_width()/2,h+5,aList[0][i],fontsize=9,color='black',ha='center',va='bottom')
            ax.text(p.get_x()+p.get_width()/2,h/9,str(round(aList[1][i]/1000.0,1))+'K',fontsize=9,color='black',ha='center',va='bottom',weight='bold')
            i=i+1

def  plot_feature_importance(est:object,train_col_list:List[str],logging:bool=False,path:Optional[str]=None,top_feats:int=20) -> None:
     if hasattr(est,"best_estimator_"):
         est=est.best_estimator_
     elif hasattr(est,"feature_importances_"):
         est=est

     feats={}
     for feature, imp in zip(train_col_list,est.feature_importances_):
         feats[feature]=imp

     importances=pd.DataFrame.from_dict(feats,orient='index').rename(columns={0:"Importance"})
     importances=importances.sort_values(by="Importance",ascending=False).head(top_feats)
     importances.plot.barh(title='Feature Importance')
     plt.xlabel('Feature Imp Score')
     if logging:
         importances.to_csv(path+'feature_imp.txt')
         plt.savefig(path+'feature_imp.png')


def formatter(x):
    d={}
    d['positive']=np.where(x['actuals']==True,1,0).sum()
    d['negative']=np.where(x['actuals']==False,1,0).sum()
    return pd.Series(d,index=['positive','negative'])


def plot_coverage_chart(df=None,pdt='tv',prob_col:str='probabilities',logging:bool=False,path:Optional[str]=None) -> None:


    df['churn_decile']=pd.qcut(df[prob_col],10,labels=False)
    df['churn_decile']=df['churn_decile']+1
    df['churn_decile']=11-df['churn_decile']

    df.loc[df.churn_decile.isin([1,2,3]),'churn_level']='High Risk'
    df.loc[df.churn_decile.isin([4,5]),'churn_level']='Mid Risk'
    df.loc[df.churn_decile.isin([6,7,8,9,10]),'churn_level']='Low Risk'

    df=df.groupby('churn_decile').apply(formatter).reset_index()

    chart_config={
        'chartStyle':'darkgrid',
        'riskLevel':{1:'High',2:'High',3:'High',4:'Mid',5:'Mid',999:'Low'},
        'riskColourPal':{'High':(0.890196,0.101960,0.109803),'Mid':'#3778bf','Low':'#a8a495'},
        'chartTitle':'churners',
        'xlabel':"'Decile by churn'",
        'ylabel':"'No of %s churners'%(pdt.capitalize())",
        'legendTitle':"'churn'",
        'topNCoverage':5,
        'lineLabel':"'Ave No of Churners {0}%'.format(percAveChurn)"

    }
    df['decile_size']=df['positive']+df['negative']
    df['perc_positive']=((df['positive']/(df['positive']+df['negative']))*100).round(4)
    df['perc_negative']=100-df['perc_positive']

    ttlPositive=df['positive'].sum()
    df['spread_positive']=(df['positive']/ttlPositive).round(4)
    df['perc_spread_cum_positive']=df.spread_positive.cumsum(axis=0)*100
    df['churn_level']=df['churn_decile'].map(chart_config['riskLevel']).fillna(chart_config['riskLevel'][999])
    percAveChurn=round(df['perc_positive'].mean(),2)
    ave_total_churn_by_decile=df['decile_size'].mean()
    ave_churn=(percAveChurn*ave_total_churn_by_decile)/100
    sns.set_style(chart_config['chartStyle'])
    gridkw=dict(height_ratios=[0.25,0.12])
    fig,(ax1,ax2)=plt.subplots(2,1,gridspec_kw=gridkw,figsize=(9,6),frameon=False,dpi=150)
    y_max=df['positive'].max()
    ax1.set_ylim([0,roundup(y_max)])
    g=sns.catplot(x='churn_decile',y='positive',hue='churn_level',kind='bar',ax=ax1,dodge=False,legend_out=False,palette=chart_config['riskColourPal'],data=df)
    l_lineLabel=eval(chart_config['lineLabel'])

    ax1.text(8,ave_churn+5,l_lineLabel,fontsize=8,color='black',ha='left',va='bottom')
    ax1.axhline(ave_churn,color='black',linewidth=1.0,linestyle="--",alpha=0.3)
    l_topNCoverage=chart_config['topNCoverage']

    topCoverage=round((df.loc[:(l_topNCoverage-1),['positive']].sum()/df['positive'].sum())*100,2)
    print(topCoverage)
    topCoverage=int(topCoverage/10)*10
    l_chart_title=chart_config['chartTitle']
    l_xlabel=chart_config['xlabel']
    l_ylabel=eval(chart_config['ylabel'])
    l_legendTitle=eval(chart_config['legendTitle'])
    fig.suptitle(l_chart_title,fontsize=10,weight='bold',ha='center',va='bottom')
    ax1.set_xlabel(l_xlabel,fontsize=8)
    ax1.set_ylabel(l_ylabel,fontsize=8)
    ax1.tick_params(axis='both',which='major',labelsize=8)
    churnList=df['positive'].tolist()
    churn_perc_list=[str(i)+'%' for i in (df['perc_positive']).round(2).tolist()]
    display_figures(ax1,[churn_perc_list,churnList])
    leg=ax1.legend(title=l_legendTitle,loc='best',fontsize=8)
    leg.get_title().set_fontsize(8)

    cumCoverageList=[str(round(x,2))+"%" for x in df['perc_spread_cum_positive'].tolist()]
    cumCoverageList[-1]='100%'
    liftList=[str(round(churnList[0]/df['positive'].mean(),2))+'x']+['']*(len(df['churn_decile'].unique())-1)
    aTble1=ax2.table(cellText=[cumCoverageList,liftList],
                     cellLoc='center',
                     rowLabels=['Cum. Coverage','Lift'],
                     rowLoc='right',
                     loc='top',
                     bbox=[0,0.75,1,0.2])
    ax2.patch.set_visible(False)
    ax2.axis('off')
    aTble1.auto_set_font_size(False)
    aTble1.set_fontsize(9)
    for (row,col), cell in aTble1.get_celld().items():
        if(col==-1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold',size=9))
        if(row==0) and (col==4):
            cell.set_text_props(fontproperties=FontProperties(weight='bold',size=11))

    plt.tight_layout(w_pad=1)
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0.5,0.5)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    plt.savefig(path+'coverage.png')
    plt.show()

