import streamlit as st

# Define pages 
home = st.Page("pages/Automated_Invoice_Extraction.py", icon='💼')

demo_predict = st.Page("pages/Evaluate_Performance.py", icon='🎓') 

flowchart = st.Page("pages/Flowchart.py", icon='📋') # For analysis report
demo_analysis_report = st.Page("pages/Extraction_Decisions.py", icon='🧪') # Demo analysis report


# Group pages
pg = st.navigation({
    "Extraction": [home],
    "Evaluation": [demo_predict], 
    "Analysis": [flowchart, demo_analysis_report], # Grouped analysis report

})

# Run the navigation
pg.run()