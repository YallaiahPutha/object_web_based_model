import streamlit as st
from indian_currency import indian_main
from usa_currency import usa_main
from electronics import electronics_main
from animal import animals_main
from vegetables import vegetables_main
from fashionwear import fashion_main
from household import household_main
from kitchenware import kitchenware_main
from outdoor import outdoor_main
from stationary import stationary_main
from washroom import washroom_main
from fruits import  fruits_main
from US_Currency_XceptionNet import us_xception_main
from multilabel_classification import multilabel_main


st.set_page_config(
    page_title="Webpage for models",
    # page_icon="👋",
)
st.write("# NSL Light Model Library for EDGE Devices!")

st.sidebar.success("Select a demo below.")

st.markdown(
    """
  This library consists of 13 categories with a total of 226 most frequently used classes by visually impaired people. 
  This could be deployed in any edge devices. Use this interface to test it.
  """
)




def indian_currency():
    st.markdown("# Indian Currency  🇮🇳")
    st.markdown("In indian currency classification,we have only specific currency classes 10, 100 ,20 ,200 ,2000 ,50 and 500. The end output from this model is currency label names.")
    st.sidebar.markdown("# Indian Currency  🇮🇳")
    indian_main()

def usa_currency():
    st.markdown("# US Currency  🇺🇸")
    st.markdown("In US currency classification,we have only specific currency classes 100dollars, 10dollars , 1dollar ,20dollars ,50dollars and 5dollars. The end output from this model is currency label names.")
    st.sidebar.markdown("# US Currency  🇺🇸")
    usa_main()

def us_xception_currency():
    st.markdown("# US Xception Currency  🇺🇸")
    st.markdown("In US currency classification,we have only specific currency classes 100dollars, 10dollars , 1dollar ,20dollars ,50dollars and 5dollars. The end output from this model is currency label names.")
    st.sidebar.markdown("# US Xception Currency  🇺🇸")
    us_xception_main()

def electronics_object():
    st.markdown("# Electronics")
    st.markdown("In electronics detection,we have only specific electronic device classes mobilephone ,computermouse ,keyboard ,mobilecharger ,earphone ,laptop ,remote ,television ,tubelight ,watches ,wallbulb ,ceilingfan ,wallclock ,refrigerator ,fan ,tablefan and airconditioner. The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Electronics")
    electronics_main()

def animals_object():
    st.markdown("# Animals")
    st.markdown("In animal detection,we have only specific animal classes hen ,dog ,goat ,cat ,cow , buffalo ,sheep and pig. The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Animals")
    animals_main()

def vegetables_object():
    st.markdown("# Vegetables")
    st.markdown("In vegetables detection,we have only specific vegetable classes ladiesfinger ,potatoes ,tomato ,greencapsicum ,brinjal ,onions ,garlic ,greenchili ,leafyvegetables ,ridgegourd ,greenbeans ,broadbeans ,curryleaves ,bittergourd ,bottlegourd ,pumpkin ,cucumber ,radish ,gingerroot ,drumsticks ,clusterbeans ,carrot ,redchilli ,corianderleaves ,broccoli and kidneybeans. The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Vegetables")
    vegetables_main()

def fashion_object():
    st.markdown("# Fashionwear")
    st.markdown("In fashion detection,we have only specific fashion classes spectacle ,facemask ,slippers ,hairband ,helmet ,handbags ,bangle ,dresses ,comb ,hairclip ,waterbottles ,wallets ,belts ,shirts ,tshirts ,ring ,earrings ,nailpolish ,umbrellas ,socks ,caps ,jeans ,canestick ,sportsshoes and formalshoes . The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Fashionwear")
    fashion_main()
def household_object():
    st.markdown("# Household")
    st.markdown("In household detection,we have only specific household classes blanket ,bed ,chair ,cushion ,sofa ,door ,curtain ,stool ,cupboard ,switchboard ,drawer ,window ,glassbottle ,carrybag ,pillow ,sofachair ,cartonbox ,doorhandle ,glassbowl ,tvcabinet ,glass ,idol ,gaslighter ,shoestand ,diningtable ,glasstable ,ledpanellight ,wineglass ,bookshelf and handwovencot . The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Household")
    household_main()

def kitchenware_object():
    st.markdown("# Kitchenware")
    st.markdown("In kitchenware detection,we have only specific kitchenware classes foodcontainer ,cooker ,jar ,steelbowl ,pot ,spoon ,knife ,fork ,cup ,cremaicplates ,cremaicbowl ,glassbowl ,steelplates ,microwaveoven ,mixiegrinder ,tray ,fryingpan ,choppingboard ,gasstove ,lpgcylinder ,tap ,inductionstove ,thermoflask ,hotbox ,jug ,gaslighter ,lunchbox ,chapatistick ,nailcutter ,kitchencabinet and kettle . The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Kitchenware")
    kitchenware_main()

def outdoor_object():
    st.markdown("# Outdoor")
    st.markdown("In outdoor detection,we have only specific outdoor classes gate ,bicycle ,plant ,tree ,stairs ,motorcycle ,bench ,car ,trolleybag ,sculpture ,autorikshaw ,bus ,railtrack ,shoppingtrolley ,tankerlorry ,aeroplane ,manhole ,directionsignboard ,tractor ,heavytruck ,callingbell and signboard . The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Outdoor")
    outdoor_main()

def stationary_object():
    st.markdown("# Stationary")
    st.markdown("In stationary detection,we have only specific stationarys classes pen ,eraser ,protractor ,book ,tape ,scale ,paper ,keys ,calander ,penstand ,fiverupeecoin ,tworupeecoin ,stapler ,calculator ,stamppad ,binderclips ,paperclip ,paperweight ,onerupeecoin ,marker ,sketchpen ,chalkpeice ,duster ,file ,globe ,colourcrayons ,staplerpins ,tenrupeecoin and stickynotes . The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Stationary")
    stationary_main()

def washroom_object():
    st.markdown("# Washroom")
    st.markdown("In washroom detection,we have only specific washroom classes bucket ,toothbrush ,toothpaste ,washbasin ,trashcan ,mug ,soapdispenser ,westerncommode ,sink and soap. The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Washroom")
    washroom_main()
def fruit_object():
    st.markdown("# Fruits")
    st.markdown("In fruit detection,we have only specific fruit classes lemon ,cherry ,banana ,applered ,coconut ,guava,pomegranate ,orange ,papaya ,kiwi ,mosambi ,grapewhite ,grapeblue ,watermelon ,avocado ,pineapple ,custardapple ,mango ,jamunfruit ,iceapple and raspberry. The end output from this model is bounding boxes and label names.")
    st.sidebar.markdown("# Fruits")
    fruits_main()

def multilabel():
    st.markdown("# Multilabel Image Classification")
    st.markdown("In multilabel classification, we are using all classes from the above detection category classes like fruits category classes,electronic category classes etc ...")
    st.sidebar.markdown("# Multilabel Image Classification")
    multilabel_main()
# def page3():
#     st.markdown("# Page 3 🎉")
#     st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Indian Currency Classification": indian_currency,
    "US Currency Classification": usa_currency,
    "US XceptionNet Currency Classification": us_xception_currency,
    "Electronics Detection":electronics_object,
    "Animals Detection":animals_object,
    "Vegetables Detection":vegetables_object,
    "Fashionwear Detection":fashion_object,
    "Household Detection":household_object,
    "Kitchenware Detection":kitchenware_object,
    "Outdoor Detection":outdoor_object,
    "Stationary Detection":stationary_object,
    "Washroom Detection":washroom_object,
    "Fruits Detection":fruit_object,
    "Multi Label Detection": multilabel,
    # "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()