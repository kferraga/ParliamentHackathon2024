#Importing the required libraries
library(readr)
library(xml2)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(tidytext)
library(wordcloud2)
library(arrow)
library(quanteda)
library(quanteda.textplots)
library(quanteda.textstats)
library(tm)
library(topicmodels)
library(RColorBrewer)
library(stringr)
library(lubridate)
library(psych)
library(ggcorrplot)





# Setting file path to extract speeches from all GB countries 
xml_file_path <- "C:/Users/User/OneDrive - University of Helsinki/Desktop/Vaibhav_Studies/Digital Humanities Hackathon/ParlaMint-GB.TEI/ParlaMint-GB.xml"
base_dir <- "C:/Users/User/OneDrive - University of Helsinki/Desktop/Vaibhav_Studies/Digital Humanities Hackathon/ParlaMint-GB.txt/"

# Reading the xml file - ParlaMint-GB.xml which has all the 
xml_content <- read_xml(xml_file_path)
namespaces <- xml_ns(xml_content)


# Extracting the file names
file_names <- xml_find_all(xml_content, ".//xi:include", namespaces) %>%
  xml_attr("href") %>%
  .[8:length(.)] %>%
  sub("\\.xml$", ".txt", .)

# Combining all the files to get the text file for all GB speeches
combined_data <- lapply(file_names, function(file_name) {
  file_path <- file.path(base_dir, file_name)
  read_delim(file_path, delim = "\t", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
}) %>%
  do.call(rbind, .)

#Changing column names
colnames(combined_data) <- c("ID", "Speech")

#Working on extracting all the meta-data files now
metadata_filenames <- gsub(".txt", "-meta-en.tsv", file_names)

# Merge all files from file_names with their corresponding metadata files
data_list <- list()

for (i in 1:length(file_names)) {
  file_path <- file.path(base_dir, file_names[i])
  metadata_path <- file.path(base_dir, metadata_filenames[i])
  
  data <- read_delim(file_path, delim = "\t", escape_double = FALSE, col_names = FALSE, trim_ws = TRUE)
  colnames(data) <- c("ID", "Speech")
  
  metadata <- read_delim(metadata_path, delim = "\t", escape_double = FALSE, trim_ws = TRUE)
  
  merged_data <- merge(data, metadata, by = 'ID')
  
  data_list[[file_names[i]]] <- merged_data
}

# Merge all merged dataframes into one and adding the country column, removing row names
final_merged_data <- do.call(rbind, data_list)
final_merged_data$Country <- 'Great Britain'

final_merged_data <- final_merged_data %>% 
  remove_rownames

final_merged_data_uk <- final_merged_data
############################################################################################
#Same code run to extract speecehs and meta-data from Ukraine and Slovenian datasets. 
#Hence removed
# 3 countries are merged merged_df2 <- rbind(final_merged_data_uk,final_merged_data_ukr,final_merged_data_si )
############################################################################################

# Making graphs for the second presentation - metadata
#1. Faceted Country, Gender and Party Orientation


ggplot(merged_df, aes(x = Party_orientation, fill = Speaker_gender)) +
  geom_bar(position = "dodge") +
  facet_wrap(~Country) +
  labs(y = "Count", x = "Party Orientation", fill = "Gender") +
  theme_minimal()


p1 <- ggplot(df1, aes(x=year,y=gdpPercap, fill=country)) + 
  geom_bar(stat='identity', position='dodge') + # Making the bar graph 
  geom_text(aes(label=round(gdpPercap,0)),  position=position_dodge(width=10), vjust = 0.5, hjust = 1.1,angle=90, color= ifelse(df1$country == 'Bangladesh','white','black')) + # Adding text to the bars and making tilting so they fit inside bar. 
  # Using conditionals to change color so text is visible.
  #Below is a preset R graph  
  labs(
    x = "Years",
    y = "GDP Per Capita",
    title = "Trends in GDP per Capita in Selected Asian Countries (In Dollars)",
    subtitle = "Comparing India, China and Bangladesh",
    caption = "Data: Gapminder",
    fill = 'Country'
  ) +
  theme_bw()+
  scale_fill_manual(values=c('#000000','#727272','#DAA520'))+
  theme(
    axis.text.x = element_text(face='bold'),
    axis.title.x = element_text(color="black", size=10, face="bold"),
    plot.subtitle = element_text(color = 'black',size = 8, hjust = 0.5),
    axis.title.y = element_text(color="black", size=10, face="bold"),
    plot.title = element_text(color="black", size=12, hjust = 0.5),
    plot.caption = element_text(face="italic")
  )
p1


# Mutating data to make cleaner graphs as well as merging longer category names to shorter ones
merged_df <- merged_df %>%
  mutate(Speaker_gender = case_when(
    Speaker_gender %in% c("F", "Female") ~ "Female",
    Speaker_gender %in% c("M", "Male") ~ "Male",
    Speaker_gender %in% c("-", "Prefer not to Say") ~ "Prefer not to Say",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(Speaker_gender))

# Party_orientation into four categories
merged_df <- merged_df %>%
  mutate(Party_orientation = case_when(
    Party_orientation %in% c("Centre", "Centre-left", "Centre-left to left", "Centre-leftCentre-left",
                             "Centre to centre-left", "Centre;Centre-left") ~ "Centre-left",
    Party_orientation %in% c("Centre-right", "Centre-right to right", "Centre-right to right;Centre",
                             "Centre-right;Centre", "Centre-right;Left", "Centre-rightCentre-left",
                             "Centre-rightCentre to centre-left", "Centre;Centre-right", "Centre to centre-right",
                             "Centre;Centre to centre-right") ~ "Centre-right",
    Party_orientation %in% c("Left", "Far-left", "Centre;Syncretic politics", "Syncretic politics") ~ "Left",
    Party_orientation %in% c("Right", "Right to far-right") ~ "Right",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(Party_orientation))

# Some more plots
p2 <- ggplot(merged_df, aes(x = Speaker_gender, fill = Party_orientation)) +
  geom_bar(position = "dodge") +
  facet_wrap(~Country) +
  labs(
    y = "Count",
    x = "Party Orientation",
    title = "Distribution of Speaker Gender by Party Orientation and Country",
    subtitle = "Grouped by Country and Party Orientation",
    caption = "Source: Your Data Source",
    fill = "Gender"
  ) +
  theme_bw() +
  scale_fill_manual(values = c('#000000', '#727272', '#DAA520','red')) +
  theme(
    axis.text.x = element_text(face = 'bold'),
    axis.title.x = element_text(color = "black", size = 10, face = "bold"),
    plot.subtitle = element_text(color = 'black', size = 8, hjust = 0.5),
    axis.title.y = element_text(color = "black", size = 10, face = "bold"),
    plot.title = element_text(color = "black", size = 12, hjust = 0.5),
    plot.caption = element_text(face = "italic")
  )

print(p2)


#Creating some quick dataframes to assist in quick plotting and using percenatge values

df_percentage <- merged_df %>%
  group_by(Country, Speaker_gender, Party_orientation) %>%
  summarise(count = n()) %>%
  ungroup() %>%
  group_by(Country, Speaker_gender) %>%
  mutate(percentage = count / sum(count) * 100)

df_percentage <- merged_df %>%
  group_by(Country, Speaker_gender) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)

p3 <- ggplot(df_percentage, aes(x = Speaker_gender, y = percentage)) +
  geom_bar(stat = "identity", fill ='black') +
  facet_wrap(~Country) +
  labs(
    y = "Percentage",
    x = "Speaker Gender",
    title = "Distribution of Speaker Gender by Party Orientation and Country - Interested Countries",
    caption = "ParlaMint4.0",
    fill = "Party Orientation"
  ) +
  geom_text(aes(label=round(percentage, 0)), vjust = 0.5, hjust = 1.1,angle=90, color = 'white') +  
  theme_bw() +
  scale_fill_manual(values = c('#000000', '#727272', '#DAA520', 'red','purple')) +
  theme(
    axis.text.x = element_text(face = 'bold'),
    axis.title.x = element_text(color = "black", size = 10, face = "bold"),
    plot.subtitle = element_text(color = 'black', size = 8, hjust = 0.5),
    axis.title.y = element_text(color = "black", size = 10, face = "bold"),
    plot.title = element_text(color = "black", size = 12, hjust = 0.5),
    plot.caption = element_text(face = "italic")
  )

print(p3)

p4 <- ggplot(, aes(x = , y = percentage)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Country) +
  labs(
    y = "Percentage",
    x = "Speaker Gender",
    title = "Distribution of Speaker Gender by Party Orientation and Country - Interested Countries",
    caption = "ParlaMint4.0",
    fill = "Party Orientation"
  ) +
  geom_text(aes(label=round(percentage, 0)),  position=position_dodge(width=10), vjust = 0.5, hjust = 1.1,angle=90) +  
  theme_bw() +
  scale_fill_manual(values = c('#000000', '#727272', '#DAA520', 'red','purple')) +
  theme(
    axis.text.x = element_text(face = 'bold'),
    axis.title.x = element_text(color = "black", size = 10, face = "bold"),
    plot.subtitle = element_text(color = 'black', size = 8, hjust = 0.5),
    axis.title.y = element_text(color = "black", size = 10, face = "bold"),
    plot.title = element_text(color = "black", size = 12, hjust = 0.5),
    plot.caption = element_text(face = "italic")
  )

print(p4)


# Creating wordclouds and frequency tables to see the gender differences in the threats class 
# Only for the UK 
# to lowercase
uk_meta_speech$Speech <- tolower(uk_meta_speech$Speech)

# Tokenize and removing stop words
tidy_speech <- uk_meta_speech %>%
  unnest_tokens(word, Speech) %>%
  anti_join(stop_words)

#Subsetting the speeches
speech_male <- tidy_speech %>% filter(Speaker_gender == "Male")
speech_female <- tidy_speech %>% filter(Speaker_gender == "Female")

#Comparing the most spoken words among the two speakers
word_freq_male <- speech_male %>%
  count(word, sort = TRUE)

word_freq_female <- speech_female %>%
  count(word, sort = TRUE)


# Male speakers word cloud
wordcloud(words = word_freq_male$word, 
          freq = word_freq_male$n, 
          min.freq = 1, 
          max.words = 100, 
          random.order = FALSE, 
          rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))

# Female speakers word cloud
wordcloud(words = word_freq_female$word, 
          freq = word_freq_female$n, 
          min.freq = 1, 
          max.words = 100, 
          random.order = FALSE, 
          rot.per = 0.35, 
          colors = brewer.pal(8, "Set3"))


# Combing the dataframes to make one for arguement data model  
UK_data_lowercase <- read_excel("C:/Users/User/Downloads/UK-data-lowercase.xlsx", col_names = FALSE)
colnames(UK_data_lowercase) <- c('ID','text')

Slovenia_data_lowercase <- read_excel("C:/Users/User/Downloads/Slovenia-data-lowercase.xlsx", col_names = FALSE)
colnames(Slovenia_data_lowercase) <- c('ID','text')


Ukraine_data_lowercase <- read_excel("C:/Users/User/Downloads/Ukraine-data-lowercase.xlsx", col_names = FALSE)
colnames(Ukraine_data_lowercase) <- c('ID','text')

df_country_lc <- rbind(UK_data_lowercase,Slovenia_data_lowercase,Ukraine_data_lowercase)

library(arrow)
write_parquet(df_country_lc, "C:/Users/User/Downloads/arguementdata.parquet")

#Working with the predictions from the arguements data

#Combing the predictions with the metadata
argument_predictions <- read_csv("C:/Users/User/Downloads/argument_predictions.csv")
UA <- read_csv("C:/Users/User/Downloads/ParlaMint-UA-metadata.csv")
GB <- read_csv("C:/Users/User/Downloads/ParlaMint-GB-metadata.csv")
SI <- read_csv("C:/Users/User/Downloads/ParlaMint-SI-metadata.csv")

#Spliting the cols in the metadata to create a merge
argument_predictions <- argument_predictions %>%
  separate(ID, into = c("Name", "Date", "ID"), sep = ",") %>%
  mutate(Country = str_extract(ID, "(?<=ParlaMint-)[A-Z]{2}")) %>%
  mutate(name_id = paste(Country, Name, sep = "-"))

countries <- rbind(GB,UA,SI)
arguement_metadata <- merge(argument_predictions, countries)


arguement_metadata <- arguement_metadata %>%
  mutate(label_text = case_when(
    codes %in% c("LABEL_0") ~ "Value",
    codes %in% c("LABEL_1") ~ "Threat",
    codes %in% c("LABEL_2") ~ "Leverage",
    codes %in% c("LABEL_3") ~ "Others",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(codes))

# Changing python list format to make them workable in R
arguement_metadata <- arguement_metadata %>%
  mutate(party_orientation_new = gsub("\\[|\\]|'", "", party_orientation))

arguement_metadata <- arguement_metadata %>%
  mutate(parties_new = gsub("\\[|\\]|'", "", parties))

arguement_metadata <- arguement_metadata %>%
  mutate(legislative_branch_new = gsub("\\[|\\]|'", "", legislative_branch))

# Relabelling metadata
arguement_metadata <- arguement_metadata %>%
  mutate(branch = case_when(
    legislative_branch_new %in% c("Lower", "Lower house") ~ "Lower",
    legislative_branch_new %in% c("Upper") ~ "Upper",
    legislative_branch_new %in% c("Lower, Upper") ~ "Lower and Upper",
    legislative_branch_new %in% c("Unicameralism") ~ "Unicameralism",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(codes))


# Some quick statistics with the new arguements predictions
country <- arguement_metadata %>% 
  group_by(label_text, Country) %>% 
  summarise(mean = mean(probs))

gender <- arguement_metadata %>% 
  group_by(label_text, gender) %>% 
  summarise(mean = mean(probs))

branch_gender <- arguement_metadata %>% 
  group_by(gender, branch, label_text) %>% 
  summarise(count = n(),
            probs = mean(probs)) %>% 
  na.omit() %>% 
  mutate()


p5 <- ggplot(branch_gender, aes(x=branch,y=count, fill=gender)) + 
  geom_bar(stat='identity', position='dodge') + # Making the bar graph 
  geom_text(aes(label=round(count,0)),  position=position_dodge(width = 1), vjust = 0.5, hjust = 1.1,angle=90, color= 'white') + # Adding text to the bars and making tilting so they fit inside bar.+ 
  facet_wrap(~label_text) + 
  # Using conditionals to change color so text is visible.
  #Below is a preset R graph setting I copy paste everywhere :)  
  labs(
    x = "Legislative Branch",
    y = "Count",
    title = "Count of Arguement Classes by Gender and Legislative Branch",
    subtitle = "Based on UKM Ukraine and Slovenia Data",
    caption = "Data: ParlmaMint4.0",
    fill = 'Gender'
  ) +
  theme_bw()+
  scale_fill_manual(values=c('#000000','#727272','#DAA520'))+
  theme(
    axis.text.x = element_text(face='bold'),
    axis.title.x = element_text(color="black", size=10, face="bold"),
    plot.subtitle = element_text(color = 'black',size = 8, hjust = 0.5),
    axis.title.y = element_text(color="black", size=10, face="bold"),
    plot.title = element_text(color="black", size=12, hjust = 0.5),
    plot.caption = element_text(face="italic")
  )
p5



# Filtering out only high probability labels and subsetting by gender to see gender differences
high_prb <- arguement_metadata %>% 
  filter(probs > 75)

threat_male <- high_prb %>% 
  filter(gender == 'M') %>% 
  filter(label_text == 'Threat')


threat_female <- high_prb %>% 
  filter(gender == 'F') %>% 
  filter(label_text == 'Threat')

# Converting to clean corpus 
text <- threat_male$text
docs <- Corpus(VectorSource(text))
docs <- docs %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("english"))
dtm <- TermDocumentMatrix(docs) 
matrix <- as.matrix(dtm) 
words <- sort(rowSums(matrix),decreasing=TRUE) 
df <- data.frame(word = names(words),freq=words)

#Creating wordclouds for male and female
set.seed(1234)
male <- df %>% 
  filter(freq > 100)
wordcloud2(data=male, size=1.6, color='black', shape = 'sqaure')

#Wordcloud for female was created similarly

#Adding emotions to the arguement classifier
#Loading all the speech and meta data

library(tidyverse)

# Loading the emotions dataset 
anew_emotion_all <- read_csv("C:/Users/User/Downloads/anew_emotion_all.csv")


# Creating a dictionary based on the stevenson (2007) dataset and it's translations
emotional_dic <- function(df) {
  happy_dict <- anew_emotion_all %>% 
    filter(mean_hap == pmax(mean_hap, mean_ang, mean_sad, mean_fear)) %>% 
    select(english_term, slovenian_term, ukranian_term) %>% 
    unlist()  %>%
    unique()
  
  
  angry_dict <- anew_emotion_all %>% 
    filter(mean_ang == pmax(mean_hap, mean_ang, mean_sad, mean_fear)) %>% 
    select(english_term, slovenian_term, ukranian_term) %>% 
    unlist() %>%
    unique()
  
  sad_dict <- anew_emotion_all %>% 
    filter(mean_sad == pmax(mean_hap, mean_ang, mean_sad, mean_fear)) %>% 
    select(english_term, slovenian_term, ukranian_term) %>%
    unlist() %>% 
    unique()
  
  fear_dict <- anew_emotion_all %>% 
    filter(mean_fear == pmax(mean_hap, mean_ang, mean_sad, mean_fear)) %>% 
    select(english_term, slovenian_term, ukranian_term) %>%
    unlist() %>%
    unique()
  
  list(happy = happy_dict, angry = angry_dict, sad = sad_dict, fear = fear_dict)
}

dicts <- emotional_dic(words_df)

text_df <- arguement_metadata %>% 
  select(c(name_id,text))



#Converting to binary columns and running dictionary through each sentence
check_emotion <- function(text, words) {
  any(str_detect(text, paste(words, collapse = "|")))
}

text_df <- text_df %>%
  mutate(
    happy = as.integer(sapply(text, check_emotion, words = dicts$happy)),
    angry = as.integer(sapply(text, check_emotion, words = dicts$angry)),
    sad = as.integer(sapply(text, check_emotion, words = dicts$sad)),
    fear = as.integer(sapply(text, check_emotion, words = dicts$fear))
  )

#Combing both the dfs to have one df with both emotions and arguements
emotion_metadata_label <- cbind(arguement_metadata,text_df2)

#Working with additional meta-data which be useful (was not)
map_roles <- function(role_str) {
  if (grepl("minister", role_str, ignore.case = TRUE)) {
    return("minister")
  } else if (grepl("member", role_str, ignore.case = TRUE)) {
    return("member")
  } else if (grepl("head", role_str, ignore.case = TRUE)) {
    return("head")
  } else if (grepl("representative", role_str, ignore.case = TRUE)) {
    return("representative")
  } else {
    return("other")
  }
}

emotion_metadata_label$new_roles <- sapply(emotion_metadata_label$roles, map_roles)


test_df <- emotion_metadata_label %>% 
  select(ID,Country,gender,label_text,branch,new_roles )

#Manually creating binary columns for the entire dataset
#Country
test_df$GB <- ifelse(test_df$Country == 'GB', 1,0)
test_df$SI <- ifelse(test_df$Country == 'SI', 1,0)
test_df$UA <- ifelse(test_df$Country == 'UA', 1,0)


#Gender
test_df$Male <- ifelse(test_df$gender == 'M', 1,0)
test_df$Female <- ifelse(test_df$gender == 'F', 1,0)

#label text
test_df$Leverage <- ifelse(test_df$label_text == 'Leverage', 1,0)
test_df$Others <- ifelse(test_df$label_text == 'Others', 1,0)
test_df$Threat <- ifelse(test_df$label_text == 'Threat', 1,0)
test_df$Value <- ifelse(test_df$label_text == 'Value', 1,0)


#branch 
test_df$Lower <- ifelse(test_df$branch == 'Lower', 1,0)
test_df$LandU <- ifelse(test_df$branch == 'Lower and Upper', 1,0)
test_df$Unicameralism <- ifelse(test_df$branch == 'Unicameralism', 1,0)
test_df$Upper <- ifelse(test_df$branch == 'Upper', 1,0)


#newroles
test_df$head <- ifelse(test_df$new_roles == 'head', 1,0)
test_df$member <- ifelse(test_df$new_roles == 'member', 1,0)
test_df$minister <- ifelse(test_df$new_roles == 'minister', 1,0)
test_df$other_role <- ifelse(test_df$new_roles == 'other', 1,0)


### Merging the emotions arguements dataset wit the metadata with the binary dataframe to perform quick analysis 
colnames(final_binary_df)[1] <- 'nameid'
colnames(final_binary_df)[4] <- 'ID2'
colnames(final_binary_df)[5] <- 'text2'
colnames(final_binary_df)[16] <- 'label_text2'
final_binary_df <- cbind(emotion_metadata_label, test_df)

#Removing duplicate/not-required columns from df
final_binary_df_all <- final_binary_df %>% 
  select(name_id, Name, Date, ID2, text2, probs, term_start, parties_new, party_orientation_new,label_text2, happy, angry, sad, fear, sum_columns,GB,SI,UA,Male,Female,Leverage,Others,Threat,Value,Lower,LandU,Unicameralism,Upper,head,member,minister,other_role )



# Calculating the parliment experience at the time of speech for further analysis
final_binary_df_all$experience <- as.period(interval(final_binary_df_all$term_start, final_binary_df_all$Date))$year

final_binary_df_all <- final_binary_df_all %>% 
  filter(experience >= -1)

# Reading the final data parquet file send by niklas to look at some interesting correlations
final_df <- read_parquet("C:/Users/User/Downloads/final_data.parquet")

final_df2 <- final_df %>% 
  select(c(happy:`Far-right`))

#Converting expercience to binary as well based on the dsitribution
ggplot(final_df2, aes(x=experience))+ 
  geom_histogram(bins = 40)+ 
  scale_x_continuous(breaks = c(1:60))


final_df2$exp_high <- ifelse(final_df2$experience > 5, 1,0)
final_df$experience <- ifelse(final_df$experience > 5, 1,0)



# Since the df is binary, selecting only binary columns and doing a tetrachoric correlation
# Made changes to the above code to look at different correlation matrices
final_df3 <- final_df %>% 
  select(c(happy:fear, Leverage:Value, GB:UA))


tab2 <-tetrachoric(final_df3)
df_cor <- as.data.frame(tab2[["rho"]])

ggcorrplot(df_cor, type = "lower",outline.col = 'black')

ggcorrplot(df_cor,
           type = "lower",
           outline.col = "black",
           ggtheme = ggthemes::theme_clean(),
           colors = c("red",'#FAF9F6',"blue"),
           title = 'Correlations Between Arguements and Emotions') 



################################################################################################

#Final Edits - 26th May 2024

################################################################################################

