Search.setIndex({docnames:["automl_infrastructure","automl_infrastructure.classifiers","automl_infrastructure.classifiers.adapters","automl_infrastructure.experiment","automl_infrastructure.experiment.metrics","automl_infrastructure.experiment.observations","automl_infrastructure.interpretation","automl_infrastructure.interpretation.lime","automl_infrastructure.pipeline","automl_infrastructure.pipeline.steps","automl_infrastructure.utils","automl_infrastructure.visualization","index","modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":2,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,sphinx:56},filenames:["automl_infrastructure.rst","automl_infrastructure.classifiers.rst","automl_infrastructure.classifiers.adapters.rst","automl_infrastructure.experiment.rst","automl_infrastructure.experiment.metrics.rst","automl_infrastructure.experiment.observations.rst","automl_infrastructure.interpretation.rst","automl_infrastructure.interpretation.lime.rst","automl_infrastructure.pipeline.rst","automl_infrastructure.pipeline.steps.rst","automl_infrastructure.utils.rst","automl_infrastructure.visualization.rst","index.rst","modules.rst"],objects:{"":{automl_infrastructure:[0,0,0,"-"]},"automl_infrastructure.classifiers":{adapters:[2,0,0,"-"],base:[1,0,0,"-"],ensemble_classifier:[1,0,0,"-"]},"automl_infrastructure.classifiers.adapters":{keras_classifier:[2,0,0,"-"],sklearn:[2,0,0,"-"]},"automl_infrastructure.classifiers.adapters.keras_classifier":{KerasClassifierAdapter:[2,1,1,""]},"automl_infrastructure.classifiers.adapters.keras_classifier.KerasClassifierAdapter":{get_params:[2,2,1,""],set_params:[2,2,1,""]},"automl_infrastructure.classifiers.adapters.sklearn":{SklearnClassifierAdapter:[2,1,1,""]},"automl_infrastructure.classifiers.adapters.sklearn.SklearnClassifierAdapter":{get_params:[2,2,1,""],set_params:[2,2,1,""]},"automl_infrastructure.classifiers.base":{BasicClassifier:[1,1,1,""],Classifier:[1,1,1,""],ClassifierPrediction:[1,1,1,""]},"automl_infrastructure.classifiers.base.BasicClassifier":{fit:[1,2,1,""],predict:[1,2,1,""],predict_proba:[1,2,1,""]},"automl_infrastructure.classifiers.base.Classifier":{fit:[1,2,1,""],get_params:[1,2,1,""],name:[1,2,1,""],predict:[1,2,1,""],predict_proba:[1,2,1,""],set_params:[1,2,1,""]},"automl_infrastructure.classifiers.base.ClassifierPrediction":{classes_pred:[1,2,1,""],classes_proba:[1,2,1,""]},"automl_infrastructure.classifiers.ensemble_classifier":{EnsembleClassifier:[1,1,1,""]},"automl_infrastructure.classifiers.ensemble_classifier.EnsembleClassifier":{fit:[1,2,1,""],get_params:[1,2,1,""],predict:[1,2,1,""],predict_proba:[1,2,1,""],set_params:[1,2,1,""]},"automl_infrastructure.experiment":{base:[3,0,0,"-"],metrics:[4,0,0,"-"],observations:[5,0,0,"-"],params:[3,0,0,"-"]},"automl_infrastructure.experiment.base":{Experiment:[3,1,1,""]},"automl_infrastructure.experiment.base.Experiment":{X:[3,2,1,""],add_observation:[3,2,1,""],add_visualization:[3,2,1,""],additional_training_data_X:[3,2,1,""],additional_training_data_y:[3,2,1,""],best_model:[3,2,1,""],dump:[3,2,1,""],end_time:[3,2,1,""],get_model_observations:[3,2,1,""],get_model_visualizations:[3,2,1,""],load:[3,2,1,""],objective_name:[3,2,1,""],objective_score:[3,2,1,""],print_report:[3,2,1,""],refresh:[3,2,1,""],remove_visualization:[3,2,1,""],run:[3,2,1,""],y:[3,2,1,""]},"automl_infrastructure.experiment.metrics":{base:[4,0,0,"-"],max_recall_at_precision:[4,0,0,"-"],standard_metrics:[4,0,0,"-"],threshold_min_precision:[4,0,0,"-"],utils:[4,0,0,"-"]},"automl_infrastructure.experiment.metrics.base":{Metric:[4,1,1,""],SimpleMetric:[4,1,1,""]},"automl_infrastructure.experiment.metrics.base.Metric":{measure:[4,2,1,""]},"automl_infrastructure.experiment.metrics.base.SimpleMetric":{measure:[4,2,1,""],measure_lst:[4,2,1,""]},"automl_infrastructure.experiment.metrics.max_recall_at_precision":{MaxRecallAtPrecision:[4,1,1,""]},"automl_infrastructure.experiment.metrics.max_recall_at_precision.MaxRecallAtPrecision":{measure_lst:[4,2,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics":{Accuracy:[4,1,1,""],CohenKappa:[4,1,1,""],F1Score:[4,1,1,""],MetricFactory:[4,1,1,""],ObjectiveFactory:[4,1,1,""],Precision:[4,1,1,""],Recall:[4,1,1,""],Support:[4,1,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.Accuracy":{measure:[4,2,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.CohenKappa":{measure:[4,2,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.F1Score":{measure:[4,2,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.MetricFactory":{create:[4,2,1,""],standard_metrics:[4,3,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.ObjectiveFactory":{create:[4,2,1,""],standard_objectives:[4,3,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.Precision":{measure:[4,2,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.Recall":{measure:[4,2,1,""]},"automl_infrastructure.experiment.metrics.standard_metrics.Support":{measure:[4,2,1,""]},"automl_infrastructure.experiment.metrics.threshold_min_precision":{ThresholdMinPrecision:[4,1,1,""]},"automl_infrastructure.experiment.metrics.threshold_min_precision.ThresholdMinPrecision":{measure_lst:[4,2,1,""]},"automl_infrastructure.experiment.metrics.utils":{parse_metric:[4,4,1,""],parse_objective:[4,4,1,""]},"automl_infrastructure.experiment.observations":{base:[5,0,0,"-"],standard_observations:[5,0,0,"-"]},"automl_infrastructure.experiment.observations.base":{Observation:[5,1,1,""],SimpleObservation:[5,1,1,""]},"automl_infrastructure.experiment.observations.base.Observation":{observe:[5,2,1,""]},"automl_infrastructure.experiment.observations.base.SimpleObservation":{agg_func:[5,2,1,""],observe:[5,2,1,""]},"automl_infrastructure.experiment.observations.standard_observations":{Avg:[5,1,1,""],Std:[5,1,1,""]},"automl_infrastructure.experiment.observations.standard_observations.Avg":{agg_func:[5,2,1,""]},"automl_infrastructure.experiment.observations.standard_observations.Std":{agg_func:[5,2,1,""]},"automl_infrastructure.experiment.params":{ListParameter:[3,1,1,""],OptunaParameterSuggester:[3,1,1,""],Parameter:[3,1,1,""],ParameterSuggester:[3,1,1,""],RangedParameter:[3,1,1,""]},"automl_infrastructure.experiment.params.ListParameter":{copy:[3,2,1,""],suggest:[3,2,1,""]},"automl_infrastructure.experiment.params.OptunaParameterSuggester":{suggest_continuous_float:[3,2,1,""],suggest_discrete_float:[3,2,1,""],suggest_int:[3,2,1,""],suggest_list:[3,2,1,""]},"automl_infrastructure.experiment.params.Parameter":{copy:[3,2,1,""],name:[3,2,1,""],set_name:[3,2,1,""],suggest:[3,2,1,""]},"automl_infrastructure.experiment.params.ParameterSuggester":{suggest_continuous_float:[3,2,1,""],suggest_discrete_float:[3,2,1,""],suggest_int:[3,2,1,""],suggest_list:[3,2,1,""]},"automl_infrastructure.experiment.params.RangedParameter":{copy:[3,2,1,""],suggest:[3,2,1,""]},"automl_infrastructure.interpretation":{lime:[7,0,0,"-"],permutation_importance:[6,0,0,"-"]},"automl_infrastructure.interpretation.lime":{lime_graph:[7,0,0,"-"]},"automl_infrastructure.interpretation.lime.lime_graph":{GraphDomainMapper:[7,1,1,""],LimeGraphExplainer:[7,1,1,""]},"automl_infrastructure.interpretation.lime.lime_graph.GraphDomainMapper":{map_exp_ids:[7,2,1,""]},"automl_infrastructure.interpretation.lime.lime_graph.LimeGraphExplainer":{explain_instance:[7,2,1,""]},"automl_infrastructure.interpretation.permutation_importance":{PermutationImportance:[6,1,1,""]},"automl_infrastructure.interpretation.permutation_importance.PermutationImportance":{fit:[6,2,1,""],show_weights:[6,2,1,""]},"automl_infrastructure.pipeline":{base:[8,0,0,"-"],steps:[9,0,0,"-"]},"automl_infrastructure.pipeline.base":{Pipeline:[8,1,1,""]},"automl_infrastructure.pipeline.base.Pipeline":{fit:[8,2,1,""],get_params:[8,2,1,""],last_step:[8,2,1,""],predict:[8,2,1,""],predict_proba:[8,2,1,""],set_params:[8,2,1,""],steps:[8,2,1,""]},"automl_infrastructure.pipeline.steps":{base:[9,0,0,"-"]},"automl_infrastructure.pipeline.steps.base":{GenericStep:[9,1,1,""],Step:[9,1,1,""]},"automl_infrastructure.pipeline.steps.base.GenericStep":{fit:[9,2,1,""],name:[9,2,1,""],set_params:[9,2,1,""],transform:[9,2,1,""],transformer:[9,2,1,""]},"automl_infrastructure.pipeline.steps.base.Step":{fit:[9,2,1,""],get_params:[9,2,1,""],name:[9,2,1,""],set_params:[9,2,1,""],transform:[9,2,1,""],transformer:[9,2,1,""]},"automl_infrastructure.utils":{functions:[10,0,0,"-"]},"automl_infrastructure.utils.functions":{extract_ordered_classes:[10,4,1,""],random_str:[10,4,1,""]},"automl_infrastructure.visualization":{base:[11,0,0,"-"],confusion_matrix:[11,0,0,"-"],precision_recall_curve:[11,0,0,"-"]},"automl_infrastructure.visualization.base":{Visualization:[11,1,1,""]},"automl_infrastructure.visualization.base.Visualization":{fit:[11,2,1,""],show:[11,2,1,""]},"automl_infrastructure.visualization.confusion_matrix":{ConfusionMatrix:[11,1,1,""]},"automl_infrastructure.visualization.confusion_matrix.ConfusionMatrix":{fit:[11,2,1,""],repredict_other_label:[11,2,1,""],show:[11,2,1,""],to_dict:[11,2,1,""]},"automl_infrastructure.visualization.precision_recall_curve":{PrecisionRecallCurve:[11,1,1,""]},"automl_infrastructure.visualization.precision_recall_curve.PrecisionRecallCurve":{classes_:[11,2,1,""],fit:[11,2,1,""],get_curve:[11,2,1,""],show:[11,2,1,""]},automl_infrastructure:{classifiers:[1,0,0,"-"],experiment:[3,0,0,"-"],interpretation:[6,0,0,"-"],pipeline:[8,0,0,"-"],utils:[10,0,0,"-"],visualization:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"abstract":[1,3,4,5,11],"class":[1,2,3,4,5,6,7,8,9,11],"default":3,"final":3,"function":3,"int":[1,3],"return":[1,2,3,4,5,7,8,10],"static":[3,4],"true":[1,2,3,4,5,8,11],"try":3,"while":[1,2,8],For:5,Its:4,The:[1,2,3,4,5,8],With:[],abc:[1,3,4,5,11],accord:4,accuraci:[3,4,5,6],adapt:1,add:3,add_dat:3,add_observ:3,add_visu:3,added:3,addit:3,additional_training_data_i:3,additional_training_data_x:3,agg_func:5,aggreg:[1,3,4,5],all:[3,4,5],along:4,alphabet:[1,4,8],also:[1,3,8],among:3,ani:[1,2,3,4,8],api:3,appli:1,arg:7,arrai:[1,3,8],ask:4,auto:[1,7],automl_infrastructur:12,averag:[4,5],avg:5,base:[2,6,7],basicclassifi:[1,2],best:3,best_model:3,between:1,bool:[1,2,3,4,8],bound:3,built:3,calcul:[3,4],callabl:[3,4,5],can:3,charset:10,choos:3,class_nam:11,classes_:11,classes_pr:1,classes_proba:1,classifi:[0,3,4,5,8],classifier_predict:4,classifier_prediction_lst:[5,10,11],classifierpredict:[1,4,5],close:[3,4],cohen:4,cohen_kappa:4,cohenkappa:4,column:[3,5],common:1,complex:[1,2,3,8],concept:5,concret:4,confus:3,confusion_matrix:[],confusionmatrix:[3,11],consist:[1,3],contain:[1,3,8],content:12,continu:3,contract:[1,2],copi:3,core:3,cosin:7,coupl:1,creat:[3,4],cross:[3,5],curv:4,custom:3,custom_threshold:11,data:3,datafram:[1,3,5,8],dataset:[1,8],decim:3,declar:3,deep:[1,2,3,8],defin:[1,3,4,5],demand:[],deviat:5,dict:[1,2,3,8],dictionari:[1,2,3,4,8],discret:3,displai:3,distance_metr:7,divid:3,domainmapp:7,dump:3,dure:[3,4],each:[1,3,4,5,8],ect:[1,3,5],embedding_col:7,encod:2,encode_label:2,end:3,end_tim:3,ensembl:1,ensemble_classifi:[],ensemble_extra_featur:1,ensemble_model:1,ensembleclassifi:1,estim:6,evalu:3,everi:[1,3,4,5],examin:3,exp:7,experi:[0,1],explain_inst:7,explan:7,extra:1,extract_ordered_class:10,extrem:5,f1_score:4,f1score:4,factori:4,fals:[2,3,4,7],father:3,featur:[1,2,3,8],feature_select:7,features_col:[1,2,8],features_df:7,figsiz:11,find:4,fit:[1,6,8,9,11],fold:[3,5],formal:4,found:3,from:[1,3,4],full:3,futur:4,gather:1,gener:5,genericstep:9,get:[3,5],get_curv:11,get_model_observ:3,get_model_visu:3,get_param:[1,2,8,9],getter:[1,2,8],give:5,given:[1,3,4,5,8,10],goal:4,graph:7,graphdomainmapp:7,group:[3,4],has:4,have:[3,5],hierarchi:3,high:3,higher:3,how:4,hyper:[1,2,3,8],hyper_paramet:3,idx:7,ignor:[1,2,8],implement:[1,2,3,4,5],includ:3,index:12,indic:7,initi:[3,5,7],inner:[1,2,8],input_model:1,instanc:[3,4,5],interfac:[1,3,4],interpret:0,is_group:4,iter:3,its:3,itself:3,jupyt:3,kappa:4,kera:2,keras_classifi:[],kerasclassifieradapt:2,kernel:7,kernel_width:7,know:[],kwarg:[1,7,8,9],label:[1,2,3,4,5,7,8],last_step:8,leav:5,length:10,librari:[],lime:6,lime_graph:[],limegraphexplain:7,linear:4,linear_cohen_kappa:4,list:[1,2,3,4,5,7,8],listparamet:3,load:3,log:3,look:3,low:3,lower:3,lr1:[1,2,8],mai:[1,3,4,5,8],main:[1,4],mani:4,map:7,map_exp_id:7,match:2,matrix:3,max_classes_per_figur:11,max_recall_at_precis:[],maxim:3,maximum:4,maxrecallatprecis:4,mean:5,measur:4,measure_lst:4,method:[1,3,4,5,8],metric:[1,3,5],metricfactori:4,minim:3,model:[1,2,3,4,7],model_nam:3,modul:12,more:1,must:[1,3,4,5,8],n_fold:3,n_iter:6,n_job:3,n_repetit:3,n_threshold:11,n_trial:3,name:[1,2,3,4,5,7,8,9],narrow:1,nativ:[1,8],natur:4,neighbor:7,neighbors_lst:7,node:7,node_col:7,node_nam:7,non:[4,5],none:[1,2,3,6,7,11],normal:11,note:[1,3,4,5,8],notebook:3,num_featur:7,num_sampl:7,number:[3,4,5],numpi:[1,3,8],object:[1,3,4,6,7,9],objective_nam:3,objective_scor:3,objectivefactori:4,observ:3,observation_typ:3,one:[1,3,4,5],oper:1,optim:3,option:[1,2,3,4,5,8],optuna:3,optunaparametersuggest:3,optunaparametersuggestor:[],order:[1,4,8,11],other:[1,2,3,8,11],other_class:11,otherwis:4,output:5,output_class_col:5,output_observation_col:5,own:3,packag:[12,13],page:12,panda:[1,3,4,5,8],parallel:3,param:[1,2,8],paramet:[1,2,3,4,5,8],parametersuggest:3,parametersuggestor:[],parent_model:3,pars:4,parse_metr:4,parse_object:4,part:5,partli:4,path:3,perform:3,permutation_import:[],permutationimport:6,pick:3,pipelin:0,possibl:[1,3],potenti:5,pre:4,precis:[3,4],precision_recall_curv:[],precisionrecallcurv:11,predefin:3,predict:[1,4,5,8],predict_proba:[1,8],print:3,print_func:3,print_report:3,probabl:[1,4,8],probs_lst:11,process:3,properti:[1,3,8,9,11],quadrat:4,quadratic_cohen_kappa:4,random:10,random_st:[6,7],random_str:10,rang:3,rangedparamet:3,rather:4,reason:1,recal:4,receiv:[1,4,5,8],refresh:3,regener:3,remov:3,remove_visu:3,repeat:3,report:3,repredict_other_label:11,repres:[1,3],requir:[3,4],row:[1,8],run:3,same:4,sample_weight:11,scale:3,score:[3,4,5,6],search:12,select:4,seri:[1,3,4,5,8],set:[1,2,3,5,8],set_nam:3,set_param:[1,2,8,9],setup:[],should:[1,2,5,8],show:11,show_weight:6,shown:3,simplemetr:4,simpleobserv:5,singl:[4,5],size:[3,4],sklearn:[],sklearn_model:2,sklearnclassifieradapt:2,some:3,sort:[1,3,4,8],speak:4,specif:1,split:3,standard:5,standard_metr:[],standard_object:4,standard_observ:[],start:3,std:[3,5],step:[3,8],step_rat:3,str:[1,2,3,4,5,8],string:[1,4,7,8,10],sub:[1,3,4,5],sub_model1:3,sub_model2:3,submodul:[],subpackag:[],suggest:3,suggest_continous_float:[],suggest_continuous_float:3,suggest_discrete_float:3,suggest_int:3,suggest_list:3,suggestor:[],summari:3,suppli:3,support:[1,3,4],suppos:5,test:3,than:4,them:[1,5],thi:3,thing:3,threshold:4,threshold_min_precis:[],thresholdminprecis:4,time:3,to_dict:11,togeth:3,top:[1,5],top_label:7,train:[1,3,8],transform:9,transformer_class:9,trial:3,tupl:7,type:[1,3,4,8],typic:1,under:3,understand:3,unifi:4,unrol:1,upon:3,upper:3,use:[3,4],used:3,useful:5,user:[3,4],using:4,util:0,valid:[3,5],valu:[1,2,3,4,5,8],vector:[1,3,8],verbos:7,vertic:7,visual:[0,3],wai:1,want:[3,5],wasn:4,watch:3,weather:[1,2,3,4,8],weight:[4,7],weighted_f1_scor:4,weighted_precis:4,weighted_recal:4,when:5,which:3,wide:1,won:3,work:[1,2,8],worker:3,wrap:[2,5],wrapper:[2,3],y_pred:1,y_proba:1,y_true:4,y_true_lst:[5,10,11],you:[3,5]},titles:["automl_infrastructure package","automl_infrastructure.classifiers package","automl_infrastructure.classifiers.adapters package","automl_infrastructure.experiment package","automl_infrastructure.experiment.metrics package","automl_infrastructure.experiment.observations package","automl_infrastructure.interpretation package","automl_infrastructure.interpretation.lime package","automl_infrastructure.pipeline package","automl_infrastructure.pipeline.steps package","automl_infrastructure.utils package","automl_infrastructure.visualization package","Welcome to AutoML Infrastructure\u2019s documentation!","automl_infrastructure"],titleterms:{"function":10,adapt:2,automl:12,automl_infrastructur:[0,1,2,3,4,5,6,7,8,9,10,11,13],base:[1,3,4,5,8,9,11],classifi:[1,2],confusion_matrix:11,content:[0,1,2,3,4,5,6,7,8,9,10,11],document:12,ensemble_classifi:1,experi:[3,4,5],indic:12,infrastructur:12,interpret:[6,7],keras_classifi:2,lime:7,lime_graph:7,max_recall_at_precis:4,metric:4,modul:[0,1,2,3,4,5,6,7,8,9,10,11],observ:5,packag:[0,1,2,3,4,5,6,7,8,9,10,11],param:3,permutation_import:6,pipelin:[8,9],precision_recall_curv:11,setup:[],sklearn:2,standard_metr:4,standard_observ:5,step:9,submodul:[1,2,3,4,5,6,7,8,9,10,11],subpackag:[0,1,3,6,8],tabl:12,threshold_min_precis:4,util:[4,10],visual:11,welcom:12}})