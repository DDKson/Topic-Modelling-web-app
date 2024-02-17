import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import re
import pyLDAvis
import pyLDAvis.gensim
from streamlit import components
from data_crawler.Shopee_crawl import ShopeeCrawler
from unsupervised_models.unsupervised import UnsupervisedModels

st.set_page_config(layout="wide")
st.title(f"Topic Analysis")
df = None 
input_option = st.sidebar.radio("Input option: ", ["Upload data", "Product Link"])
if df is None:
    if input_option == "Upload data":
        uploaded_file = st.sidebar.file_uploader("Upload your review data here", type = [".csv"])
        if uploaded_file:
            @st.cache_data
            def get_data_from_file(uploaded_file):
                df = pd.read_csv(uploaded_file)
                df = df.dropna(subset = "comment").reset_index()
                return df
            df = get_data_from_file(uploaded_file)
    else:
        crawler = ShopeeCrawler()
        link = st.sidebar.text_input("$\\textsf{\Large Enter Product Link}$")

        if link:
            @st.cache_data
            def get_data_from_link():
                shop_id, item_id = crawler.get_ids_from_link(link)
                data = crawler.Crawl(item_id, shop_id)
                df = pd.DataFrame(data)
                return df
            data = get_data_from_link()
            df = data.copy()
            df = df.dropna(subset = "comment").reset_index()
            download_content = df.to_csv(encoding = "utf-16")
            st.sidebar.download_button(label="Download data as csv file",
                data=download_content,
                file_name="data.csv",
            )
reset = st.sidebar.button("Reset")
if reset:
    df = None
    st.cache_data.clear()
    st.cache_resource.clear()

mode_option = st.sidebar.radio("Mode option: ", ["Topic Discovery", "Topic Classification"])
if df is not None:
    topic_num = st.sidebar.number_input("Number of topics", step = 1, min_value = 2, max_value = len(df), format = "%d")
    if mode_option == "Topic Discovery":
        df = df.dropna(axis = 0, subset = ["comment"])
        @st.cache_resource
        def train(topic_num):
            lda_model = UnsupervisedModels(data = df, model = "LDA")
            # This function will only be run the first time it's called
            trained_model = lda_model.train_gensim_models(num_topics = topic_num, evaluate = True)
            vis = pyLDAvis.gensim.prepare(trained_model, lda_model.corpus, lda_model.id2word, sort_topics=False)
            html_string = pyLDAvis.prepared_data_to_html(vis)
            return (lda_model, trained_model, html_string)

        if topic_num:
            lda_model, trained_model, html_string = train(topic_num)
            st.markdown(f"### Coherence Score: {round(lda_model.evaluation['Coherence Score'], 2)}, Topic Diversity: {round(lda_model.evaluation['Topic Diversity'], 2)}")
            col1, col2 = st.columns((2, 1))
            with col1:
                components.v1.html(html_string, width=1500, height=800)

            def format_topics_sentences(ldamodel, corpus, texts):
                sent_topics_df = pd.DataFrame()
                rows = []
                for i, row in enumerate(ldamodel[corpus]):
                    row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
                    for j, (topic_num, prop_topic) in enumerate(row):
                        if j == 0:
                            rows.append(pd.Series([int(topic_num), round(prop_topic,4)]))
                        else:
                            break
                sent_topics_df = pd.DataFrame(rows)
                sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution']
                sent_topics_df["original"] = texts
                return(sent_topics_df)
            topic_df = format_topics_sentences(trained_model, lda_model.corpus, lda_model.clean_data["comment"].values)
            topic_df["Dominant_Topic"] += 1

            with col2:
                topic_counts = topic_df["Dominant_Topic"].value_counts().sort_index()
                topic_counts = pd.DataFrame({"Topic": topic_counts.index, "Count": topic_counts.values})
                topic_counts["Topic"] = "Topic " + topic_counts["Topic"].astype(int).astype(str)
                fig = go.Figure(
                    data=[go.Pie(
                        labels=topic_counts["Topic"],
                        values=topic_counts["Count"],
                        sort=False
                        )
                    ])
                fig.update_layout(autosize=True, width=500, height=800, title={
                    'text': "Topic Distribution",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                    legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5
                ))
                st.plotly_chart(fig)

            st.markdown("# Individual review analysis")
            col1, col2 = st.columns((1, 2), gap = "medium")
            col2_1, col2_2 = col2.columns(2)
            n = col2_1.number_input("Select which comment to analyze", step = 1, min_value = 0, max_value = len(topic_df), format = "%d")
            inference_result = trained_model[lda_model.corpus[n]]

            topic_higlight = col2_1.selectbox('Which topic to highlight',
    ["Topic " + str(i[0] + 1) for i in inference_result[0]])
            topic_higlight = int(re.search(r"\d+$", topic_higlight).group())
            select_row = topic_df.iloc[int(n)]
            col1.dataframe(topic_df[['Dominant_Topic', "original"]])
            col2_1.markdown(f"**Highlighting words contributing to :red[Topic {topic_higlight}] in review #{n}:**")
            
            topic_contrib_data = pd.DataFrame({"Topic": ["Topic " + str(i[0] + 1) for i in inference_result[0]], "Percent Contribution": [i[1] for i in inference_result[0]]})
            fig = px.bar(topic_contrib_data, y='Percent Contribution', x='Topic', title=f'Weight of each topic in review #{n}', width=600, height=400)
            col2_2.plotly_chart(fig)
            def replace_colored(text, lda_output, current_topic, color):
                dominant_word = lda_output[1]
                words = []
                for word_id, topic in dominant_word:
                    if current_topic in topic:
                        word = trained_model.id2word[word_id]
                        words.append(word)
                if words:
                    pattern = r"\b(?<!\w)(?P<word>{})\b".format("|".join(words))
                    return re.sub(pattern, r":{}[\g<word>]".format(color), text, flags=re.IGNORECASE)
                return text
            
            text = select_row["original"].replace("\n", "  \n  ").replace(":", ": ")
            text = replace_colored(text, inference_result, topic_higlight - 1, "red")
            col2_1.write(text)


    else:
        pass

else: 
    st.text("Select data on sidebar to get started")
    st.text("Zoom out your browser for better experience")

