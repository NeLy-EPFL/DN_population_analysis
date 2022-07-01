source("utils.R")


row.order <- read.csv(file="output/Fig2/order_behaviour_encoding.csv")
row.order <- rename_flies(row.order)

df <- read.csv("output/FigS2/predict_activity_walkingVShind.csv")
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_flies(df)
df <- rename_behaviours(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/FigS2/panel_a.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Uniquely explained neural activity variance (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Beh")

df <- read.csv("output/FigS2/predict_activity_grooming.csv")
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_flies(df)
df <- rename_behaviours(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/FigS2/panel_b.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Uniquely explained neural activity variance (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Beh")
