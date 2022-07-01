source("utils.R")


row.order <- read.csv(file="output/Fig2/order_behaviour_encoding.csv")
row.order <- rename_flies(row.order)

df <- read.csv("output/Fig2/behavior_prediction_results.csv")
df <- df[df$Context == "all", ]
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- df[c("Fly", "ROI", "Regressor", "Trial", "Fold", "rsquared")]
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_behaviours(df)
df <- rename_flies(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/Fig2/panel_b.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Variance uniquely explained by neural activity (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Behavior")
