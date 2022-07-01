source("utils.R")

row.order <- read.csv("output/Fig2/order_behaviour_encoding.csv")
row.order <- rename_flies(row.order)

df <- read.csv("output/Fig2/behavior_prediction_results.csv")
df <- df[df$Variable == "walking", ]
df <- subset(df, select=-Variable)
colnames(df)[colnames(df) == "Context"] <- "Regressor"
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_behaviours(df)
df <- rename_flies(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
df[df$rsquared_median < 0, "rsquared_median"] <- 0
output_file <- "output/FigS9/panel_a.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Variance uniquely explained by neural activity (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Behavior")

df <- read.csv("output/Fig2/behavior_prediction_results.csv")
df <- df[df$Variable == "head_grooming", ]
df <- subset(df, select=-Variable)
colnames(df)[colnames(df) == "Context"] <- "Regressor"
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_behaviours(df)
df <- rename_flies(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
df[df$rsquared_median < 0, "rsquared_median"] <- 0
output_file <- "output/FigS9/panel_b.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Variance uniquely explained by neural activity (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Behavior")
