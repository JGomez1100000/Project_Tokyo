#--------------------LIBRERÍAS----------------------------#
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from scipy.stats import zscore
# import folium
# from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap
import plotly.colors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud

#------------------------------------------------#
#tenemos dos opciones de layout, wide or centered

st.set_page_config(page_title='Tokyo Airbnb Dataset', page_icon=':izakaya_lantern:', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)


#--------------------COSAS QUE VAMOS A USAR EN TODA LA APP----------------------------#
listings_detailed_raw = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\data\listings_detailed.csv')
listings_raw = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\data\listings.csv')
listings = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\data_cleaned\listings_tokyo.csv')
reviews_details = pd.read_csv(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\data_cleaned\reviews_details_tokyo.csv')
#--------------------CONFIGURACIÓN DE LA PÁGINA----------------------------#
st.title('Tokyo Airbnb Dataset :izakaya_lantern:')

#--------------------Imagenes----------------------------#
image = Image.open(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\Streamlit\Images\mtfuji.jpg')
image2 = Image.open(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\Streamlit\Images\mtfuji2.jpg')
image3 = Image.open(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\Streamlit\Images\prefecture_mapjpg.jpg')
image4 = Image.open(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\Streamlit\Images\bandera.jpg')
image5 = Image.open(r'C:\Users\Javi\Desktop\Bootcamp2\Proyecto_modulo_2\Streamlit\Images\hiroshi.png')
#--------------------Links----------------------------#
link = '<iframe title="dashboard_airbnb_tokyo" width="1140" height="541.25" src="https://app.fabric.microsoft.com/reportEmbed?reportId=9396b597-3eba-4262-bc18-295004ae53e6&autoAuth=true&ctid=8aebddb6-3418-43a1-a255-b964186ecc64" frameborder="0" allowFullScreen="true"></iframe>'

#--------------------SIDEBAR----------------------------#
st.sidebar.image(image4, caption='Japan flag', width= 200)
st.sidebar.markdown("<h1 style='text-align: left; font-size: 40px;'>Apartados</h1>", unsafe_allow_html=True)
selection = st.sidebar.radio('', ['Introducción','Preprocesamiento', 'EDA', 'Relaciones', 'Dashboard', 'Conclusiones'])

#--------------------Introducción----------------------------#

if selection == "Introducción":
    
    st.header("Introducción")
    st.write('<b>Tokyo</b> es una de las ciudades más grandes del mundo con alrededor de 13 millones de habitantes en la zona central y 37 en toda el area metropolitana.', unsafe_allow_html=True)
    st.write('En cuanto al turismo, es una de las ciudades más visitadas del mundo por turistas entrando en el top 10 con alrededor de 13 millones de turistas extranjeros en 2022', unsafe_allow_html=True)
    st.write('De igual manera, la seguridad es uno de los pilares en japón, siendo <b>Tokyo</b> una de las ciudades más seguras del mundo.', unsafe_allow_html=True)
    st.write('Todo esto, hace de <b>Tokyo</b> una ciudad ideal para desarrollar el sector turístico.', unsafe_allow_html=True)
    st.image(image, caption='Tokyo skycrappers and Mt. Fuji')
    st.write('https://es.statista.com/estadisticas/487739/ranking-de-las-ciudades-con-mayor-numero-de-visitas-de-turistas-extranjeros/#:~:text=Ranking%20de%20las%20ciudades%20m%C3%A1s,en%20el%20mundo%20en%202022&text=Este%20ranking%20presenta%20las%20ciudades,una%20lista%20encabezada%20por%20Bangkok.', unsafe_allow_html=True)
    st.write('https://www.lainformacion.com/management/paises-mundo-considerados-mas-seguros-vivir-2022/2873944/', unsafe_allow_html=True)


#st.table(data=listings)
#--------------------Preprocesamiento----------------------------#

if selection == "Preprocesamiento":
    st.header("Preprocesamiento")
    st.write('', unsafe_allow_html=True)
    st.table(listings_raw.head())
    st.code('''listings_raw.info()''')
    st.code('''listings_raw.isnull().sum()''')
    st.code(listings_raw.isnull().sum())
    st.write('<b>Quitamos la columna neighbourhood_group ya que tiene todos los valores nulos</b>', unsafe_allow_html=True)
    st.code('''del listings['neighbourhood_group']''')
    st.subheader('License')
    st.write('<b>Quitamos los valores nulos ya que esas filas tienen más valores nulos y no afectan al dataset</b>', unsafe_allow_html=True)
    st.code('''listings = listings.dropna(subset=['license']''')
    st.write('<b>Los valores están bastante mal distribuidos, por lo que vamos a agruparlos</b>', unsafe_allow_html=True)
    st.code(listings_raw['license'].value_counts())
    st.write('<b>Los valores están bastante mal distribuidos, por lo que vamos a agruparlos</b>', unsafe_allow_html=True)
    st.code('''license_hotel = 'Hotels and Inns Business Act',
        license_special = 'Special Economic Zoning Act',
        license_M1300 = 'M1300',
        license_m1300 = 'm1300',
        license_M5400 = 'M5400' ''')
    st.code('''(listings_raw.loc[listings_raw['license'].str.contains(license_hotel), 'license'] = license_hotel),
        (listings.loc[listings['license'].str.contains(license_special), 'license'] = license_special),
        (listings.loc[listings['license'].str.contains(license_M1300), 'license'] = license_M1300),
        (listings.loc[listings['license'].str.contains(license_m1300), 'license'] = license_M1300),
        (listings.loc[listings['license'].str.contains(license_M5400), 'license'] = license_M5400),
        (listings.loc[listings['license'].str.contains('30新保衛環第126号'), 'license'] = license_special)''')
    st.code('''listings['license'].value_counts()''')
    st.code(listings['license'].value_counts())
    st.subheader('Buscamos las columnas que pueden servirnos de listings_details')
    st.code('''target_columns = ['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_identity_verified', 'property_type', 'accommodates', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'instant_bookable']''')
    st.write('<b>Creamos un dataframe con esas columnas y lo concatenamos con el dataframe listings</b>', unsafe_allow_html=True)
    st.code('''columns_df = pd.DataFrame(listings_details[target_columns])''')
    st.code('''listings = pd.concat([listings, columns_df], axis=1)''')

    st.write('<b>Tratamos las columnas con valores nulos, formato erróneo y encodeamos</b>', unsafe_allow_html=True)
    st.code('''listings['host_response_rate'] = listings['host_response_rate'].str.replace('%', '')
        listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.replace('%', '')
        listings['host_response_rate'] = listings['host_response_rate'].astype(float)
        listings['host_acceptance_rate'] = listings['host_acceptance_rate'].astype(float)''')
    st.code('''median_response = listings['host_response_rate'].median()
        median_acceptance = listings['host_acceptance_rate'].median()
        listings['host_response_rate'] = listings['host_response_rate'].fillna(median_response)
        listings['host_acceptance_rate'] = listings['host_acceptance_rate'].fillna(median_acceptance)''')
    st.subheader('Property_type')
    st.code('''listings_detailed_raw['property_type'].value_counts()''')
    st.code(listings_detailed_raw['property_type'].value_counts())

  
    st.write('<b>Arreglamos la columna Property_type agrupando.</b>', unsafe_allow_html=True)
    st.code(''' property_entire = 'Entire'
        property_private_room = 'Private room'
        property_share_room = 'Shared room'
        property_room = 'Room' ''')
    st.code('''listings.loc[listings['property_type'].str.contains(property_entire), 'property_type'] = property_entire
        listings.loc[listings['property_type'].str.contains(property_private_room), 'property_type'] = property_private_room
        listings.loc[listings['property_type'].str.contains(property_share_room), 'property_type'] = property_share_room
        listings.loc[listings['property_type'].str.contains(property_room), 'property_type'] = property_room''')
    st.code(listings['property_type'].value_counts())
    st.subheader('Encodes:')
    st.write('''

license_encode

    Hotels and Inns Business Act : 0
    M1300 : 1
    Special Economic Zoning Act : 2
    M5400 : 3

room_type_encode

    Entire home/apt : 0
    Private room' : 1
    Shared room' : 2
    Hotel room' : 3

neighbourhoods_encode

    0-40

host_response_time_encode

    within an hour' : 0
    within a few hours: 1
    within a day : 2
    a few days or more : 3

host_is_superhost_encode

    0:False
    1:True

host_identity_verified_encode

    1:True
    0:False


property_type_encode

    'Entire' : 0
    'Room' : 1
    'Private room' : 2
    'Shared room' : 3
    'Hut' : 4
    
instant_bookable_encode

    1:True
    0:False
    ''')

#--------------------Distribución de las variables----------------------------#

if selection == 'EDA':
    st.header('EDA')
    
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(['Corr', 'Price', 'Reviews per month','Neighbourhood',
                                                                           'Location','Room_type','Property type','Accommodates',
                                                                           'host_id','Hiroshi'])
                                                                           
    with tab1:
        plt.figure(figsize=(40, 25))

        columns_corr = ['price', 'latitude', 'longitude', 'neighbourhood_encode', 'host_is_superhost_encode',
        'host_identity_verified_encode', 'property_type_encode',
        'instant_bookable_encode', 'accommodates', 'review_scores_rating',
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value','license_encode',
        'room_type_encode', 'host_response_time', 'host_response_rate',
        'host_acceptance_rate','reviews_per_month',
        'calculated_host_listings_count', 'availability_365','number_of_reviews']
        listings_columns= listings[columns_corr]
        matrix_corr = listings_columns.corr()

        sns.heatmap(matrix_corr, cmap='coolwarm', vmin=-1, annot=True)
        
        st.pyplot()

    with tab2:
        st.subheader("Price")
        st.subheader('Calculamos outliers')
        st.write('<b>Calculamos el Z-score para saber cuantos valores están por encima de 2.5 std</b>', unsafe_allow_html=True)

        st.code('''column_zscores = zscore(listings['price'])   
        outliers = listings[(column_zscores > 2.5) | (column_zscores < -2.5)]   
        outliers.value_counts().sum()''')
        st.code(41)

        fig = px.histogram(listings_raw,
                        x= 'price',
                        nbins=30,
                        marginal='box',
                        title ='',
                        template= "plotly_dark")

        fig.update_layout(height=600, width=1000)
        
        st.plotly_chart(fig)

        st.write('<b>Calculamos la función para sustituir los outliers por el IQR', unsafe_allow_html=True)

        st.code('''def reemplazar_outliers_iqr(listings, columnas):
        
        for columna in columnas:
            Q1 = listings[columna].quantile(0.25)
            Q3 = listings[columna].quantile(0.75)
            IQR = Q3 - Q1
            outliers = listings[(listings[columna] < (Q1 - 1.5 * IQR)) | (listings[columna] > (Q3 + 1.5 * IQR))]
            listings[columna] = np.where(listings[columna] < (Q1 - 2.5 * IQR), Q1, listings[columna])
            listings[columna] = np.where(listings[columna] > (Q3 + 2.5 * IQR), Q3, listings[columna])''')
        st.code('''reemplazar_outliers_iqr(listings, ['price'])''')

        fig = px.histogram(listings,
                        x= 'price',
                        nbins=30,
                        marginal='box',
                        title ='',
                        template= "plotly_dark")

        fig.update_layout(height=600, width=1000)
        
        st.plotly_chart(fig)

    with tab3:
        st.subheader("Reviews per month")

        fig = px.histogram(listings_raw,
                        x= 'reviews_per_month',
                        nbins=30,
                        marginal='box',
                        title ='Distribución de reviews_per_month',
                        template= "plotly_dark")

        fig.update_layout(height=600, width=1000)

        st.plotly_chart(fig)

        st.code('''reemplazar_outliers_iqr(listings, ['reviews_per_month'])''')
        fig = px.histogram(listings,
                        x= 'reviews_per_month',
                        nbins=30,
                        marginal='box',
                        title ='Distribución de reviews_per_month',
                        template= "plotly_dark")

        fig.update_layout(height=600, width=1000)

        st.plotly_chart(fig)

    with tab4:
        fig = px.histogram(listings['neighbourhood'],
                    title='Distribución de neighbourhood',
                    template='plotly_dark')

        fig.update_layout(xaxis_tickfont_size=14)
        fig.update_layout(xaxis_title='Neighbourhood')
        fig.update_layout(yaxis_tickfont_size=14)
        fig.update_layout(yaxis_title='Cantidad')
        fig.update_layout(showlegend=False)
        fig.update_layout(
                width=1000, 
                height=700
        )

        st.plotly_chart(fig)

    with tab5:
        fig = px.scatter_mapbox(listings,
                    lat='latitude',
                    lon='longitude',
                    size='price',
                    zoom=10,
                    mapbox_style='carto-positron',
                    title='AirBnb Apartment Distribution in Tokyo per room_type',
                    template= "plotly_dark",
                    size_max=5,
                    animation_frame='room_type')
        
        st.plotly_chart(fig)

        st.map(data=listings, zoom=9, use_container_width=True)


    with tab6:
        fig = px.histogram(listings,'room_type',
                template= 'plotly_dark',
                title = 'Types of room in Tokyo')
        
        fig.update_layout(xaxis_tickfont_size=14)
        fig.update_layout(xaxis_title='Types of room')
        fig.update_layout(yaxis_tickfont_size=14)
        fig.update_layout(yaxis_title='Cantidad')
        fig.update_layout(showlegend=False)
        fig.update_layout(
                width=1000, 
                height=700
        )
        st.plotly_chart(fig)

    with tab7:

        prop = listings.groupby(['property_type','room_type']).room_type.count()
        prop = prop.unstack()
        fig = px.bar(prop,
            barmode="group",
            color_discrete_map={"Entire home/apt":"red","Private room":"yellow","Shared room":"green"},
            template = "plotly_dark")
        
        fig.update_layout(xaxis_tickfont_size=14)
        fig.update_layout(xaxis_title='Types of room')
        fig.update_layout(yaxis_tickfont_size=14)
        fig.update_layout(yaxis_title='Cantidad')
        fig.update_layout(
                width=1000, 
                height=700
        )

        st.plotly_chart(fig)

    with tab8:

        feq = listings['accommodates'].value_counts().sort_index()
        num_categories = len(feq)
        #Creamos un conjunto de colores para que se asignen a cada categoría, chatgpt help
        colors = plotly.colors.qualitative.Dark24[:num_categories]

        fig = px.bar(feq,
                title='Cantidad de personas por alojamiento en Tokyo',
                template="plotly_dark",
                labels=dict(index="Accommodates", value="Listings"),
                width=1000,
                color=colors,
                )


        st.plotly_chart(fig)

    with tab9:
        host_quan = listings['host_id'].value_counts().sort_index()
        fig = px.histogram(host_quan,
                    marginal='box',
                    template='plotly_dark')
        
        fig.update_layout(xaxis_tickfont_size=14)
        fig.update_layout(xaxis_title='Types of room')
        fig.update_layout(yaxis_tickfont_size=14)
        fig.update_layout(yaxis_title='Cantidad')
        fig.update_layout(
                width=1000, 
                height=700
        )

        st.plotly_chart(fig)

    with tab10:
        host_private = listings.groupby(['host_id', 'host_name', 'room_type', 'property_type']).size().reset_index(name='listings').sort_values(by='listings', ascending=False)
        st.code('''host_private = listings.groupby(['host_id', 'host_name', 'room_type', 'property_type']).size().reset_index(name='listings').sort_values(by='listings', ascending=False)''')
        st.code('''host_private.head(20)''')
        st.code(host_private.head(20))
        st.write('<b> Invesigamos al host Hiroshi, ya que al parecer tiene 98 listings</b>', unsafe_allow_html=True)
        st.code(host_private[host_private['host_id'] ==258668827.0])
        st.write('<b> Vemos que se trata de un hotel capsula</b>', unsafe_allow_html=True)
        hotel_hiroshi = listings[listings['room_type'] == 'Hotel room']

        hiroshi = hotel_hiroshi[hotel_hiroshi['host_id']== 258668827.0]
        hiroshi = hiroshi[['name','host_id', 'host_name', 'latitude', 'longitude']]
        hiroshi.index.name = "listing_id"
        hiroshi

        st.image(image5, caption='')




#--------------------Relaciones----------------------------#

if selection == "Relaciones":
    st.header("Relaciones")

    tab11, tab12, tab13, tab14, tab15, tab16, tab17= st.tabs(['Precio', 'Neighbourhoods in Tokyo', 'Superhost',
                                                                                    'Host','Rating/Superhost','Instant bookable','Words'])
    
    with tab11:
        precio = listings.groupby('neighbourhood')['price'].mean().sort_values(ascending = True)
        dfprecio = pd.DataFrame(precio)
        dfprecio = dfprecio.reset_index()
     #ahora graficamos con plotly
        fig = px.area(dfprecio, 
            x='neighbourhood',
            y='price',template = 'plotly_dark',
            title = 'Average daily price based on location in Tokyo')
    
        fig.update_layout(
                width=1000, 
                height=700)
        
        st.plotly_chart(fig)

        fig = px.treemap(dfprecio,path=['neighbourhood'],
           values='price',
           template= 'plotly_dark',
           title='Average daily price based on location in Tokyo' )
        fig.update_layout(
                width=1000, 
                height=700)
        st.plotly_chart(fig)

    with tab12:
        feq = listings.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)

        tokyo_neigh = gpd.read_file(r'C:\Users\Javi\Desktop\Bootcamp\Proyecto_modulo_2\data\neighbourhoods.geojson')
        # Asegurarse de que 'feq' es un DataFrame de pandas correcto antes de transponerlo
        # Si 'feq' es un DataFrame, no necesitamos encapsularlo en una lista y convertirlo a DataFrame otra vez
        feq = feq.transpose()
        # Mezclamos
        tokyo_neigh = pd.merge(tokyo_neigh, feq, on='neighbourhood', how='left')

        # Renombramos la columna
        tokyo_neigh.rename(columns={'price': 'average_price'}, inplace=True)

        # Redondeamos el precio
        tokyo_neigh.average_price = tokyo_neigh.average_price.round(decimals=2)

        # Conseguimos colores para nuestras casas 
        map_dict = tokyo_neigh.set_index('neighbourhood')['average_price'].to_dict()
        color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))

        def get_color(feature):
            value = map_dict.get(feature['properties']['neighbourhood'])
            if value is not None:
                return color_scale(value)
            else:
                return 'grey'  # Un color predeterminado si el barrio no está en el diccionario
        
        fig = px.choropleth_mapbox(dfprecio,
                            geojson=tokyo_neigh,
                            featureidkey='properties.neighbourhood',
                            locations ="neighbourhood",
                            color = 'price',
                            color_continuous_scale="portland",
                            title="Neighbourhoods in Tokyo",
                            zoom=9, hover_data = ['neighbourhood','price'],
                            mapbox_style="carto-positron",
                            width=1000,
                            height=750,
                            center = {"lat": 35.65, "lon": 139.50}
                                )

        fig.update(layout_coloraxis_showscale=True)
        fig.update_layout( paper_bgcolor="#1f2630",font_color="white",title_font_size=20, title_x = 0.5)
        st.plotly_chart(fig)

        listings_filtered = listings[listings['neighbourhood'] == 'Kodaira Shi'][['host_id', 'host_name', 'neighbourhood', 'price']]

        st.write(listings_filtered)

    with tab13:
        listings.host_is_superhost = listings.host_is_superhost.replace({"t": "True", "f": "False"})
        feq=listings['host_is_superhost'].value_counts()

        fig = px.pie(feq,
        names = feq.index, 
        values=feq.values,
        hover_name=feq,
        width = 1000, 
        template= "plotly_dark", 
        title="Number of listings with Superhost")

        st.plotly_chart(fig)

    with tab14:
        fig = px.histogram(listings,
             x = 'host_response_rate',
             nbins=200,
           
             title='Host response rate',
             labels=dict(index='Average review score', value='Number of listings'),
             template='plotly_dark'
             )

        fig.update_xaxes(tickvals=[60,70,80,90,100], range=[89,101])
        fig.update_yaxes(title_font=dict(size=14))

        st.plotly_chart(fig)

        feq2 = listings['host_response_time'].value_counts()

        fig = px.bar(feq2,     
                    title='Host response time',
                    labels=dict(index='Average review score', value='Number of listings'),
                    template='plotly_dark'
                    )

        fig.update_xaxes(tickvals=[0,1,2,3], range=[-0.5,3.5])
        fig.update_yaxes(title_font=dict(size=14))

        st.plotly_chart(fig)

        feq2 = listings['host_acceptance_rate'].value_counts()

        fig = px.bar(feq2,     
                    title='host_acceptance_rate',
                    labels=dict(index='Host acceptance rate', value='Number of listings'),
                    template='plotly_dark'
                    )

        fig.update_xaxes(tickvals=[80,90,100], range=[79,101])
        fig.update_yaxes(title_font=dict(size=14))

        st.plotly_chart(fig)

    with tab15:

        fig = px.histogram(listings,
                 x = 'review_scores_rating',
                 facet_col= 'host_is_superhost',
                 nbins=30,
                
                 title= 'Rating score by superhost',
                 template='plotly_dark'
                )
        fig.update_layout(
                width=1000, 
                height=700)

        st.plotly_chart(fig)

        review_superhost_mean_true = listings.loc[listings['host_is_superhost']=='True', 'review_scores_rating'].mean()
        review_superhost_mean_false = listings.loc[listings['host_is_superhost']=='False', 'review_scores_rating'].mean()

        st.write('La media cuando es Superhost es:', 4.8049, ' cuando no lo es:', 4.5848)

    with tab16:
        book_response_mean = listings.groupby('instant_bookable')['host_response_time'].mean()
        num_categories = len(book_response_mean)
        #Creamos un conjunto de colores para que se asignen a cada categoría, chatgpt help
        colors = plotly.colors.qualitative.Dark24[:num_categories]

        fig = px.bar(book_response_mean,
                    title='Mean response time per instant bookable',
                    template="plotly_dark",
                    labels=dict(index="Instant bookable", value="Time"),
                    width=1000,
                    color=colors,
             )
        st.plotly_chart(fig)


    with tab17:
        st.code(reviews_details.head())
        host_reviews = reviews_details.groupby(['host_id', 'host_name']).size().sort_values(ascending=False).to_frame(name = "number_of_reviews")
        st.code(host_reviews.head())

        
        nltk.download('stopwords')

        #take out empty comments (530)
        reviews_details = reviews_details[reviews_details['comments'].notnull()]

        #remove numbers
        reviews_details['comments'] = reviews_details['comments'].str.replace('\d+', '') 
        #all to lowercase
        reviews_details['comments'] = reviews_details['comments'].str.lower()
        #remove windows new line
        reviews_details['comments'] = reviews_details['comments'].str.replace('\r\n', "")
        #remove stopwords (from nltk library)

        stop_english = stopwords.words("english")
        reviews_details['comments'] = reviews_details['comments'].apply(lambda x: " ".join([i for i in x.split() 
                                                            if i not in (stop_english)]))
        # remove punctuation
        reviews_details['comments'] = reviews_details['comments'].str.replace('[^\w\s]'," ")
        # replace x spaces by one space
        reviews_details['comments'] = reviews_details['comments'].str.replace('\s+', ' ')

        reviews_details.comments.values[2] #print same comments again

        texts = reviews_details.comments.tolist()
        vec = CountVectorizer().fit(texts)
        bag_of_words = vec.transform(texts)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

        cvec_df = pd.DataFrame.from_records(words_freq, columns= ['words', 'counts']).sort_values(by="counts", ascending=False)
        st.code(cvec_df.head(10))

        st.write('Eliminamos br')

        st.code('''cvec_df = cvec_df.drop(cvec_df[cvec_df['words'] == 'br'].index''')


        cvec_df = cvec_df.drop(cvec_df[cvec_df['words'] == 'br'].index)

        cvec_dict = dict(zip(cvec_df.words, cvec_df.counts))
        wordcloud = WordCloud(width=800, height=400)
        wordcloud.generate_from_frequencies(frequencies=cvec_dict)
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)


#--------------------Dashboard----------------------------#


if selection == "Dashboard":
    st.header("Dashboard")
    st.markdown(link, unsafe_allow_html=True)






#--------------------Conclusiones----------------------------#

if selection == "Conclusiones":
    st.header("Conclusiones")
    st.image(image2, caption='Imperial palace and Mt. Fuji')

