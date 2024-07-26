
import streamlit as st
import webbrowser
import pandas as pd
#import streamlit_pandas as sp
from PIL import Image
import plotly.express as px
from streamlit_option_menu import option_menu
import numpy as np
from datetime import datetime
import pickle

#set up page configuration for streamlit
st.set_page_config(page_title='HDB Flat Resale Price Predictor',page_icon='house',initial_sidebar_state='expanded',
                        layout='wide',menu_items={"about":'This streamlit application was developed by M.Gokul'})

#css for page setup
st.markdown("""
    <style>
    .main-menu {font-size:20px; margin-top:-40px;}
    .content {padding: 20px;}
    .header {margin-top: 20px; padding-top: 30px; text-align: center; background-color:#002b36 ; padding-bottom: 10px;}
    </style>
    """, unsafe_allow_html=True)

#df=pd.read_csv(r"C:\Users\ADMIN\Videos\capstion_project\singapure\new_singapur.csv").drop(columns="Unnamed: 0")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  
        options=["Home", "Get Prediction"], 
        
        default_index=0, 
        orientation="vertical",
        styles={
            "container": {"padding": "5px","border": "2px ridge ", "background-color": "#002b36"},
            "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px", "--hover-color": "#586e75"},
            "nav-link-selected": {"background-color":"#247579"},  
                })


if selected == "Home":
    # Welcome message
    st.markdown('## :green[Welcome to the Home Page]')

    # Form section
    with st.form(key='form', clear_on_submit=False):
        st.markdown('# :blue[Project Title]')
        st.markdown('### Singapore Resale Flat Prices Predicting')

        st.markdown('## :blue[Skills Takeaway From This Project]')
        st.markdown('### - Data Wrangling, EDA, Model Building, Model Deployment')

        st.markdown('## :blue[Domain]')
        st.markdown('### Real Estate')

        st.markdown('## :blue[Problem]')
        st.markdown(
            """
            ### The objective of this project is to develop a machine learning model and 
            ### deploy it as a user-friendly web application that predicts the resale prices of
            ### flats in Singapore. This predictive model will be based on historical data of
            ### resale flat transactions, and it aims to assist both potential buyers and
            ### sellers in estimating the resale value of a flat.
            """
        )

        # GitHub link button
        if st.form_submit_button('Visit GitHub Repository'):
            link = "https://github.com/Muthukumar0908/Industrial-Copper-Modeling.git"
            webbrowser.open_new_tab(link)


#user input values for selectbox and encoded for respective features
class option:

    option_months = ["January","February","March","April","May","June","July","August","September","October","November","December"]

    encoded_month= {"January" : 1,"February" : 2,"March" : 3,"April" : 4,"May" : 5,"June" : 6,"July" : 7,"August" : 8,"September" : 9,
            "October" : 10 ,"November" : 11,"December" : 12}

    option_town=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH','BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI',
        'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG','SERANGOON',
        'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS','PUNGGOL']
    
    encoded_town={'ANG MO KIO' : 0 ,'BEDOK' : 1,'BISHAN' : 2,'BUKIT BATOK' : 3,'BUKIT MERAH' : 4,'BUKIT PANJANG' : 5,'BUKIT TIMAH' : 6,
        'CENTRAL AREA' : 7,'CHOA CHU KANG' : 8,'CLEMENTI' : 9,'GEYLANG' : 10,'HOUGANG' : 11,'JURONG EAST' : 12,'JURONG WEST' : 13,
        'KALLANG/WHAMPOA' : 14,'LIM CHU KANG' : 15,'MARINE PARADE' : 16,'PASIR RIS' : 17,'PUNGGOL' : 18,'QUEENSTOWN' : 19,
        'SEMBAWANG' : 20,'SENGKANG' : 21,'SERANGOON' : 22,'TAMPINES' : 23,'TOA PAYOH' : 24,'WOODLANDS' : 25,'YISHUN' : 26}
    
    option_flat_type=['1 ROOM', '2 ROOM','3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION']

    encoded_flat_type={'1 ROOM': 0,'2 ROOM' : 1,'3 ROOM' : 2,'4 ROOM' : 3,'5 ROOM' : 4,'EXECUTIVE' : 5,'MULTI-GENERATION' : 6}

    option_flat_model=['2-ROOM','3GEN','ADJOINED FLAT', 'APARTMENT' ,'DBSS','IMPROVED' ,'IMPROVED-MAISONETTE', 'MAISONETTE',
                    'MODEL A', 'MODEL A-MAISONETTE','MODEL A2' ,'MULTI GENERATION' ,'NEW GENERATION', 'PREMIUM APARTMENT',
                    'PREMIUM APARTMENT LOFT', 'PREMIUM MAISONETTE','SIMPLIFIED', 'STANDARD','TERRACE','TYPE S1','TYPE S2']

    encoded_flat_model={'2-ROOM' : 0,'3GEN' : 1,'ADJOINED FLAT' : 2,'APARTMENT' : 3,'DBSS' : 4,'IMPROVED' : 5,'IMPROVED-MAISONETTE' : 6,
                'MAISONETTE' : 7,'MODEL A' : 8,'MODEL A-MAISONETTE' : 9,'MODEL A2': 10,'MULTI GENERATION' : 11,'NEW GENERATION' : 12,
                'PREMIUM APARTMENT' : 13,'PREMIUM APARTMENT LOFT' : 14,'PREMIUM MAISONETTE' : 15,'SIMPLIFIED' : 16,'STANDARD' : 17,
                'TERRACE' : 18,'TYPE S1' : 19,'TYPE S2' : 20}
    
#set up information for the 'get prediction' menu
if selected == "Get Prediction":
    st.write('')
    st.markdown("<h5 style=color:orange>To Predict the Resale Price of a Flat, Please Provide the Following Information:",unsafe_allow_html=True)
    st.write('')

    # creted form to get the user input 
    with st.form('prediction'):
        col1,col2=st.columns(2)
        with col1:

            user_month=st.selectbox(label='Month',options=option.option_months,index=None)

            user_town=st.selectbox(label='Town',options=option.option_town,index=None)

            user_flat_type=st.selectbox(label='Flat Type',options=option.option_flat_type,index=None)

            user_flat_model=st.selectbox(label='Flat Model',options=option.option_flat_model,index=None)

            floor_area_sqm=st.number_input(label='Floor area sqm',min_value=10.0)

            price_per_sqm=st.number_input(label='Price Per sqm',min_value=100.00)

        with col2:

            year=st.text_input(label='year',max_chars=4)

            block=st.text_input(label='Block',max_chars=3)

            lease_commence_date=st.text_input(label='Year of lease commence',max_chars=4)

            remaining_lease=st.number_input(label='Remaining lease year',min_value=0,max_value=99)

            years_holding=st.number_input(label='Years Holding',min_value=0,max_value=99)

            c1,c2=st.columns(2)
            with c1:
                storey_start=st.number_input(label='Storey start',min_value=1,max_value=50)
            with c2:
                storey_end=st.number_input(label='Storey end',min_value=1,max_value=51)
            
            st.markdown('<br>', unsafe_allow_html=True)

            button=st.form_submit_button('PREDICT',use_container_width=True)
   if button:
        with st.spinner("Predicting..."):

            #check whether user fill all required fields
            if not all([user_month,user_town,user_flat_type,user_flat_model,floor_area_sqm,price_per_sqm,year,block,
                        lease_commence_date,remaining_lease,storey_start,storey_end]):
                st.error("Please fill in all required fields.")

            else:
                #create features from user input 
                current_year=datetime.now().year

                current_remaining_lease=remaining_lease-(current_year-(int(year)))
                age_of_property=current_year-int(lease_commence_date)


                month=option.encoded_month[user_month]
                town=option.encoded_town[user_town]
                flat_type=option.encoded_flat_type[user_flat_type]
                flat_model=option.encoded_flat_model[user_flat_model]

                floor_area_sqm_log=np.log(floor_area_sqm)
                remaining_lease_log=np.log1p(remaining_lease)
                price_per_sqm_log=np.log(price_per_sqm)

                #opened pickle model and predict the resale price with user data
                with open('Decision_tree.pkl', 'rb') as files:
                    model=pickle.load(files)
                
                user_data=np.array([[ town, flat_type, block, flat_model, lease_commence_date, year,month, storey_start,
                                    storey_end,age_of_property, years_holding, current_remaining_lease, floor_area_sqm_log,
                                    remaining_lease_log, price_per_sqm_log ]])

                predict=model.predict(user_data)
                resale_price=np.exp(predict[0])

                #display the predicted selling price 
                st.subheader(f"Predicted Resale price is: :green[{resale_price:.2f}]")

