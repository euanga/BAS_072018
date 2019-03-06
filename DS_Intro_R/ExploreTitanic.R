titanic.df <- read.csv("c:\\dev\\Demos\\BAS2018\\DS_Intro_R\\titanic_raw.csv")

head(titanic.df, n=10)

dim(titanic.df)

str(titanic.df)

levels(titanic.df$embarked)

summary(titanic.df)

install.packages("dplyr")

library(dplyr)

df <- select(titanic.df, pclass, survived, -contains("name"), age:embarked) %>%
      filter(embarked != "")

df2<- filter(df, embarked != "")


df <- select(titanic.df, pclass, survived, -contains("name"), age:embarked) %>%
      filter(embarked != "") %>%
      group_by(survived) %>%
      summarise(
        avg_ticket_Price = mean(fare, na.rm = TRUE)
      )
  
df

barplot(df$avg_ticket_Price)

barplot(df$avg_ticket_Price, names.arg = df$survived)

barplot(df$avg_ticket_Price, names.arg = df$survived, col = c('red','green'))
