source("utils.R")

row.order.clustering <- read.csv("output/Fig3/order_clustering.csv")
row.order.clustering <- rename_flies(row.order.clustering)
df <- read.csv("output/Fig3/ball_rot_prediction_results.csv")
df <- df[df$Variable == "turn_l" | df$Variable == "turn_r" | df$Variable == "walking" | df$Variable == "speed", ]
df <- df[, names(df) != "Context"]
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_behaviours(df)
df <- rename_flies(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/Fig3/panel_c.pdf"
plot_matrix(df, "rsquared_mean", row.order.clustering, "Variance uniquely explained by neural activity (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Behavior", width=3, height=4)
