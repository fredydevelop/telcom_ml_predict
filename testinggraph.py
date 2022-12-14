import pandas as pd
import streamlit as st
import numpy as np
import pickle as pk
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report 
import base64
from sklearn import svm
#import seaborn as sns
import altair as alt



st.set_page_config(page_title='Predicto',layout='centered')



with st.sidebar:
    selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction","Model Performance"],icons=["cast","book","cast"],menu_icon="house",default_index=0)

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your Predictions</a>'
    return href


def eligibility_status(givendata):
    
    loaded_model=pk.load(open("singlePredict2.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction==0:
      return"Not Eligible"
    else:
      return "You are Eligible"



def main():
    st.header("Predicto")
    theReasons=[]
    #getting user input
    option = st.selectbox("what service plan is the customer on ?",("",'Prepaid'),key="service")
    if (option=='Prepaid'):
        Service=1
    else:
        Service=0

    if Service==0:
        theReasons.append("Postpaid are not eligible for a loan")
    
    option1 = st.selectbox('Did customer spend at least N200 Monthly?',("",'Yes', 'No'),key="monthlyRecharge")
    if (option1=='Yes'):
        spent_atLeast_N200_monthly_for3months=1
    else:
        spent_atLeast_N200_monthly_for3months=0

    if spent_atLeast_N200_monthly_for3months==0:
        theReasons.append("Your monthly recharge has been less than 200 naira for the past three months")
    
    
    option2 = st.selectbox('Any outstanding debt ?', ("",'Yes', 'No'),key="debt")
    if (option2=='Yes'):
        Outstanding_debt=1
    else:
        Outstanding_debt=0

    if Outstanding_debt==1:
        theReasons.append("You have an outstanding debt to pay")
    
    Main_Account=st.number_input("Main account balance")

    if Main_Account >75.0:
            theReasons.append("Your main account balance is greater than ₦75")
    
    option3 = st.selectbox("Has number been registered ?",("",'Yes', 'No'),key="registered")
    if (option3=='Yes'):
        RegisteredPhoneNumber=1
    else:
        RegisteredPhoneNumber=0

    if RegisteredPhoneNumber==0:
        theReasons.append("Your number has not been registered")

    #RegisteredPhoneNumber=st.text_input("Has number been registered ?")
    option4 = st.selectbox("Been active for three months and above ?",("",'Yes', 'No'),key="activemonths")
    if (option4=='Yes'):
        ActiveFor3monthsAndAbove=1
    else:
        ActiveFor3monthsAndAbove=0

    if ActiveFor3monthsAndAbove==0:
        theReasons.append("You have not been active for the past 3 months")
    #ActiveFor3monthsAndAbove=st.text_input("Been active for three months and above ?")
    # code for Prediction
    
    Eligible = '' #for displaying result
    Reason=""
    
    # creating a button for Prediction
    if option!="" and option1!=""  and option2!=""  and option3!=""  and option4!="" and st.button('Predict'):
        Eligible = eligibility_status([Service, spent_atLeast_N200_monthly_for3months, Outstanding_debt, Main_Account, RegisteredPhoneNumber, ActiveFor3monthsAndAbove])
        st.success(Eligible)
    
    if Eligible=="You are Eligible":
        st.balloons()

    if Eligible=="Not Eligible":
        st.write("Reasons for not been eligible:")
        j=0
        k=1
        while j< len(theReasons):
            st.write(str(k) + ". " + theReasons[j])
            j=j+1
            k=k+1
            

    



def multi(input_data):
    loaded_model=pk.load(open("multiplePredict2.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    glimpse="565dataset.csv"
    showGlimpse=pd.read_csv(glimpse)
    #showGlimpse=showGlimpse.drop(["Loan_Status"],axis=1)
    #st.dataframe(dfinput.describe())
    st.header('Dataset')
    st.markdown('Preview of the uploaded dataset')
    st.dataframe(dfinput)
    #st.markdown(dfinput.shape)
    st.write("\n")
    st.write("\n")
    

    forLoanId=dfinput["Loan_ID"]
    #dfinput.replace({"Loan_Status": {'No':0,'Yes': 1}},inplace=True)
    #dfinput.replace({"Loan_Status": {'no':0,'yes': 1}},inplace=True)
    # convert categorical columns to numerical values
    dfinput.replace({'Outstanding_debt':{'No':0,'Yes':1},'Service':{'Prepaid':1,'Postpaid':0},'ActiveFor3monthsAndAbove':{'No':0,'Yes':1},'RegisteredPhoneNumber':{'No':0,'Yes':1},'spent_atLeast_N200_monthly_for3months':{'No':0,'Yes':1}},inplace=True)
    dfinput.replace({'Outstanding_debt':{'NO':0,'YES':1},'Service':{'PREPAID':1,'POSTPAID':0},'ActiveFor3monthsAndAbove':{'NO':0,'YES':1},'RegisteredPhoneNumber':{'NO':0,'YES':1},'spent_atLeast_N200_monthly_for3months':{'NO':0,'YES':1}},inplace=True)


    dfinput.replace({'Outstanding_debt':{'no':0,'yes':1},'Service':{'prepaid':1,'postpaid':0},'ActiveFor3monthsAndAbove':{'no':0,'yes':1},'RegisteredPhoneNumber':{'no':0,'yes':1},'spent_atLeast_N200_monthly_for3months':{'no':0,'yes':1}},inplace=True)
    dfinput = dfinput.drop(columns=['Loan_ID'],axis=1)

    # separating the data and label
    showGlimpse.replace({"Loan_Status": {'No':0,'Yes': 1}},inplace=True)
    showGlimpse.replace({"Loan_Status": {'no':0,'yes': 1}},inplace=True)
    # convert categorical columns to numerical values
    showGlimpse.replace({'Outstanding_debt':{'No':0,'Yes':1},'Service':{'Prepaid':1,'Postpaid':0},'ActiveFor3monthsAndAbove':{'No':0,'Yes':1},'RegisteredPhoneNumber':{'No':0,'Yes':1},'spent_atLeast_N200_monthly_for3months':{'No':0,'Yes':1}},inplace=True)

    # convert categorical columns to numerical values
    showGlimpse.replace({'Outstanding_debt':{'no':0,'yes':1},'Service':{'prepaid':1,'postpaid':0},'ActiveFor3monthsAndAbove':{'no':0,'yes':1},'RegisteredPhoneNumber':{'no':0,'yes':1},'spent_atLeast_N200_monthly_for3months':{'no':0,'yes':1}},inplace=True)

    
    #d dropping loan _id column and loan_status  column
    #dfinput = dfinput.drop(columns=["Loan_ID"],axis=1)

    # separating the data and label
    X = showGlimpse.drop(columns=['Loan_ID','Loan_Status'],axis=1)
    Y = showGlimpse['Loan_Status']

    X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
    classifier = svm.SVC(kernel='linear')
    
    #training the support Vector Macine model
    classifier.fit(X_train,Y_train)
    X_test_prediction=classifier.predict(X_test)
    cf_matrix = confusion_matrix(Y_test, X_test_prediction)
    

    #selectionofactivity starts here
    st.write()
    st.write()
    with st.sidebar:
        predictButton=st.button("Click to Predict")
        visualizeButton=st.button("Visualize uploaded data")
        selectionList=["","confusion Matrix","Reality data vs Test result"]
        #selectionw=option_menu(menu_title=None,options=["Predict your result","Visualization","confusion Matrix"],icons=["cast","book","cast"],default_index=1, orientation="horizontal")
        st.write("\n")
        st.write("\n")

       

    if predictButton:
        prediction = loaded_model.predict(dfinput)
        interchange=[]
        for i in prediction:
            if i==0:
                newi="Not Eligible"
                interchange.append(newi)
            elif i==1:
                newi="Eligible"
                interchange.append(newi)
            
        st.subheader('**Predicted output**')
        prediction_output = pd.Series(interchange, name='loan_status')
        prediction_id = pd.Series(forLoanId)
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

        #st.dataframe(prediction)
        #visualize=st.button("Data visulization")
    if  visualizeButton:
        select=["Service","Outstanding_debt","ActiveFor3monthsAndAbove"]
        with st.sidebar:
            #names=st.selectbox("choose the column for graph",select)
            p=alt.Chart(dfinput).mark_bar().encode(x=("ActiveFor3monthsAndAbove"),y="Main_Account")
            l=alt.Chart(dfinput).mark_bar().encode(x=("Outstanding_debt"),y="Main_Account")
            w=alt.Chart(dfinput).mark_bar().encode(x=("Service"),y="Main_Account")
            #p=p.properties(width=alt.Step(170))
        st.write(p)
        st.write(l)
        



if selection=="Single Prediction":
    main()

if selection=="Multi Prediction":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("predicto")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file)
        multi(uploaded_file)
    else:
        st.info('waiting for CSV file to be uploaded.')



if selection=="Model Performance":
    
    glimpse="565dataset.csv"
    showGlimpse=pd.read_csv(glimpse)
    showGlimpse.replace({"Loan_Status": {'No':0,'Yes': 1}},inplace=True)
    showGlimpse.replace({"Loan_Status": {'no':0,'yes': 1}},inplace=True)
    showGlimpse.replace({'Outstanding_debt':{'No':0,'Yes':1},'Service':{'Prepaid':1,'Postpaid':0},'ActiveFor3monthsAndAbove':{'No':0,'Yes':1},'RegisteredPhoneNumber':{'No':0,'Yes':1},'spent_atLeast_N200_monthly_for3months':{'No':0,'Yes':1}},inplace=True)
    showGlimpse.replace({'Outstanding_debt':{'no':0,'yes':1},'Service':{'prepaid':1,'postpaid':0},'ActiveFor3monthsAndAbove':{'no':0,'yes':1},'RegisteredPhoneNumber':{'no':0,'yes':1},'spent_atLeast_N200_monthly_for3months':{'no':0,'yes':1}},inplace=True)
    X = showGlimpse.drop(columns=['Loan_ID','Loan_Status'],axis=1)
    Y = showGlimpse['Loan_Status']
    X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train,Y_train)
    X_test_prediction=classifier.predict(X_test)
    cf_matrix = confusion_matrix(Y_test, X_test_prediction)
    #selectionofactivity starts here
    st.write()
    shape=classification_report(Y_test, X_test_prediction)
    #st.title("A glimpse of predicto performance")
    st.header("1. Predicto Confusion Matrix")
    st.text(shape)
    st.write("\n")
    st.write("\n")
    st.write("\n")
        #preal = X_test_prediction.reset_index()
    Y_test = Y_test.reset_index()
    real=Y_test.drop(columns="index",axis=1)
    real=real.rename(columns={"Loan_Status":"Reality"})
    jk=pd.DataFrame(X_test_prediction,columns = ['Predicted'])
        
    checkingResult = pd.concat([real, jk], axis=1)
    #X_test_prediction.rename(columns={0:"predicted Reality"},inplace = True)
    st.header("2. The test data result after training")
    st.write(checkingResult)




