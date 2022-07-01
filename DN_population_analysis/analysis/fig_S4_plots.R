source("utils.R")


row.order <- read.csv(file="output/Fig2/order_behaviour_encoding.csv")
row.order <- rename_flies(row.order)

df <- read.csv("output/FigS4/predict_activity_from_joints.csv")
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- df[df$Regressor == "all_angles",]
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_behaviours(df)
df <- rename_flies(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/FigS4/panel_b.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Variance uniquely explained by neural activity (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Behavior")

df <- read.csv("output/FigS4/predict_activity_from_joints.csv")
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- df[df$Regressor %in% c("front_angles", "middle_angles", "hind_angles"), ]
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_behaviours(df)
df <- rename_flies(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/FigS4/panel_c.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Variance uniquely explained by neural activity (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Behavior")

df <- read.csv("output/FigS4/predict_activity_from_joints.csv")
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- df[df$Regressor %in% c("front_l_angles", "middle_l_angles", "hind_l_angles", "front_r_angles", "middle_r_angles", "hind_r_angles"), ]
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_behaviours(df)
df <- rename_flies(df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/FigS4/panel_d.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Variance uniquely explained by neural activity (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Behavior")
