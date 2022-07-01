source("utils.R")


df <- read.csv("output/Fig4/odor_regression_results.csv")
row.order <- read.csv(file="output/Fig2/order_behaviour_encoding.csv")

row.order <- rename_flies(row.order)

df[df$reduced_model_rsquared < 0, "reduced_model_rsquared"] <- 0
colnames(df)[colnames(df) == "Variable"] <- "Regressor"
df <- df[df$Regressor != "all_odor",]
df <- summarize_cv_results(df)
df["Fly_ROI"] <- paste(df$Fly, df$ROI, sep=" ")
df <- rename_flies(df)
group_vars <- c("Fly", "ROI", "Fly_ROI")
beh_r2_df <- df %>% group_by_at(group_vars) %>% summarize_at(vars(-group_cols()), mean)
beh_r2_df$Regressor <- "Behavior"
beh_r2_df$rsquared_mean <- beh_r2_df$reduced_model_rsquared_mean
df <- rbind(df, beh_r2_df)
df[df$rsquared_mean < 0, "rsquared_mean"] <- 0
output_file <- "output/Fig4/panel_b.pdf"
plot_matrix(df, "rsquared_mean", row.order, "Uniquely explained neural activity variance (%)", output_file, as_perc=TRUE, ROI_label_angle=45, range=c(0, 0.9), faceting=TRUE, y_axis_label="Odor")
