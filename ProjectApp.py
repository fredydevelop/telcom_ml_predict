import pandas as pd
import streamlit as st
import numpy as np
import pickle as pk
from streamlit_option_menu import option_menu
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report 
import base64
from sklearn import svm



st.set_page_config(page_title='Credit scoring App',layout='centered')

with st.sidebar:
    selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],menu_icon="house",default_index=0)


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href


def eligibility_status(givendata):
    
    loaded_model=pk.load(open("practice2.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction==0:
      return"Not Eligible for a loan"
    else:
      return "You are Eligible for a loan"



def main():
    st.header("Check your eligibility")
    
    #getting user input
    option = st.selectbox("what service plan is the customer on ?",("",'Prepaid', 'Postpaid'),key="service")
    if (option=='Prepaid'):
        Service="1"
    else:
        Service="0"
    option1 = st.selectbox('Did customer spend at least N200 Monthly?',("",'Yes', 'No'),key="monthlyRecharge")
    if (option1=='Yes'):
        spent_atLeast_N200_monthly_for3months="1"
    else:
        spent_atLeast_N200_monthly_for3months='0'
    
    option2 = st.selectbox('Any outstanding debt ?', ("",'Yes', 'No'),key="debt")
    if (option2=='Yes'):
        Outstanding_debt="1"
    else:
        Outstanding_debt="0"
    
    Main_Account=st.number_input("Main account balance")
    
    option3 = st.selectbox("Has number been registered ?",("",'Yes', 'No'),key="registered")
    if (option3=='Yes'):
        RegisteredPhoneNumber="1"
    else:
        RegisteredPhoneNumber="0"
    #RegisteredPhoneNumber=st.text_input("Has number been registered ?")
    option4 = st.selectbox("Been active for three months and above ?",("",'Yes', 'No'),key="activemonths")
    if (option4=='Yes'):
        ActiveFor3monthsAndAbove="1"
    else:
        ActiveFor3monthsAndAbove="0"
    #ActiveFor3monthsAndAbove=st.text_input("Been active for three months and above ?")
    # code for Prediction
    
    Eligible = ''#for displaying result
    
    # creating a button for Prediction
    if option!="" and option1!=""  and option2!=""  and option3!=""  and option4!="" and st.button('Predict'):
        Eligible = eligibility_status([Service, spent_atLeast_N200_monthly_for3months, Outstanding_debt, Main_Account, RegisteredPhoneNumber, ActiveFor3monthsAndAbove])
        st.success(Eligible)
    
    if Eligible=="You are Eligible for a loan":
        st.balloons()



def job(u):
    df = pd.read_csv(u)
    st.subheader("statistical measures")
    df.replace({"Loan_Status": {'No':0,'Yes': 1}},inplace=True)
    df.replace({"Loan_Status": {'no':0,'yes': 1}},inplace=True)
    # convert categorical columns to numerical values
    df.replace({'Outstanding_debt':{'No':0,'Yes':1},'Service':{'Prepaid':1,'Postpaid':0},'ActiveFor3monthsAndAbove':{'No':0,'Yes':1},'RegisteredPhoneNumber':{'No':0,'Yes':1},'spent_atLeast_N200_monthly_for3months':{'No':0,'Yes':1}},inplace=True)

    # convert categorical columns to numerical values
    df.replace({'Outstanding_debt':{'no':0,'yes':1},'Service':{'prepaid':1,'postpaid':0},'ActiveFor3monthsAndAbove':{'no':0,'yes':1},'RegisteredPhoneNumber':{'no':0,'yes':1},'spent_atLeast_N200_monthly_for3months':{'no':0,'yes':1}},inplace=True)
    #st.dataframe(df.groupby("Loan_Status").mean())
    visualize=st.button("Data visulization")
    if  visualize:
        visual(df)
    



def visual(yu):
    chart_data = pd.DataFrame(
    yu.iloc[:20],
    columns=["Main_Account"])
    st.bar_chart(chart_data)

    #ploting histogram
    jame = pd.DataFrame(yu[0:], columns = ["Service","Outstanding_debt","ActiveFor3monthsAndAbove","Main_Account"])
    jame.hist()
    plt.show()
    st.pyplot()

    #plotting line chat
    liney = pd.DataFrame(yu[0:], columns = ["Main_Account"])
    st.line_chart(liney)    




def multi(input_data):
    loaded_model=pk.load(open("multiplePredict.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    glimpse="565dataset.csv"
    showGlimpse=pd.read_csv(glimpse)
    #showGlimpse=showGlimpse.drop(["Loan_Status"],axis=1)
    #st.dataframe(dfinput.describe())
    st.header('1. Dataset')
    st.markdown('**1.1. Overview of the uploaded dataset**')
    st.dataframe(dfinput)

    forLoanId=dfinput["Loan_ID"]
    #dfinput.replace({"Loan_Status": {'No':0,'Yes': 1}},inplace=True)
    #dfinput.replace({"Loan_Status": {'no':0,'yes': 1}},inplace=True)
    # convert categorical columns to numerical values
    dfinput.replace({'Outstanding_debt':{'No':0,'Yes':1},'Service':{'Prepaid':1,'Postpaid':0},'ActiveFor3monthsAndAbove':{'No':0,'Yes':1},'RegisteredPhoneNumber':{'No':0,'Yes':1},'spent_atLeast_N200_monthly_for3months':{'No':0,'Yes':1}},inplace=True)

    # convert categorical columns to numerical values
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

    X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=0)
    classifier = svm.SVC(kernel='linear')
    #training the support Vector Macine model
    classifier.fit(X_train,Y_train)
    X_test_prediction=classifier.predict(X_test)
    cf_matrix = confusion_matrix(Y_test, X_test_prediction)

    #tn, fp, fn, tp = cf_matrix.ravel()

    #selectionofactivity starts here
    st.write()
    st.write()
    selectionList=["confusion Matrix","Predict","Visualization"]
    #selectionw=option_menu(menu_title=None,options=["Predict your result","Visualization","confusion Matrix"],icons=["cast","book","cast"],default_index=1, orientation="horizontal")
    st.title("2.Generate results")
    selectiond=st.selectbox("Predict, Visualize, and check confusion Matrix",selectionList)
    #st.write(Y)
    if selectiond=="confusion Matrix":
        shape=classification_report(Y_test, X_test_prediction)
        st.text(shape)
        


    if selectiond=="Predict":
        prediction = loaded_model.predict(dfinput)
        interchange=[]
        for i in prediction:
            if i==0:
                newi="Not eligible"
                interchange.append(newi)
            elif i==1:
                newi="Eligible"
                interchange.append(newi)
            
        st.subheader('**Prediction output**')
        prediction_output = pd.Series(interchange, name='loan_status')
        prediction_id = pd.Series(forLoanId)
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

        #st.dataframe(prediction)
        #visualize=st.button("Data visulization")
    if  selectiond=="Visualization":
        chart_data = pd.DataFrame(dfinput.iloc[:20],colums=["Main_Account"])
        st.bar_chart(chart_data)

    #ploting histogram
        jame = pd.DataFrame(dfinput[0:], columns = ["Service","Outstanding_debt","ActiveFor3monthsAndAbove","Main_Account"])
        jame.hist()
        plt.show()
        st.pyplot()

    #plotting line chat
        liney = pd.DataFrame(dfinput[0:], columns = ["Main_Account"])
        st.line_chart(liney) 


if selection=="Single Prediction":
    main()

if selection=="Multi Prediction":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('File Upload')
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file)
        multi(uploaded_file)
    else:
        st.info('Awaiting for CSV file to be uploaded.')

if selection=="Visualization":
    visual(uploaded_file)

if __name__ == '__job__':
    job()
#elif __name__ == '__main__':
    #main()


