library(ggplot2)
library(tidyr)
library(plyr)
library(dplyr)
library(scales)
library(grid)


behaviour_colours <- c("Walking"='#E05361', "Resting"='#B7BA78', "Head grooming"='#8A4FF7', "Front leg rubbing"='#4CC8ED', "Posterior grooming"='#F7B14F', "Undefined"="#FFFFFF")

sem <- function(x){
    sd(x) / sqrt(length(x))
}

summarize_cv_results <- function(df){
    group_vars <- c("Fly", "ROI", "Regressor")
    group_vars <- intersect(group_vars, colnames(df))
    summarized_df <- df %>% group_by_at(group_vars) %>% summarize_at(vars(-group_cols(), -Fold), list("mean"=mean, "sem"=sem, "median"=median))
}

rename_behaviours <- function(df){
    old_names <- c("rest", "walking", "forward_walking", "backward_walking", "antennal_grooming", "eye_grooming", "foreleg_grooming", "hindleg_grooming", "abdominal_grooming", "PER_event", "pushing", "CO2", "Yaw", "Pitch", "Roll", "resting", "walking", "head_grooming", "hind_grooming", "turn_r", "turn_l", "conv_vel", "conv_turn", "speed")
    old_names_conv <- paste(old_names, "conv", sep=".")
    new_names <- c("Resting", "Walking", "Forward walking", "Backward walking", "Antennal grooming", "Eye grooming", "Front leg rubbing", "Rear leg grooming", "Abdominal grooming", "Proboscis extension", "Pushing", "CO\u2082 puff", "Yaw", "Pitch", "Roll", "Resting", "Walking", "Head grooming", "Posterior grooming", "Right turning", "Left turning", "Forward walking speed", "Turning angular velocity", "Forward walking speed")
    for (i in 1:length(old_names)){
        names(df)[names(df) == old_names[i]] <- new_names[i]
        names(df)[names(df) == old_names_conv[i]] <- new_names[i]
    }
    if ("Behaviour" %in% names(df)){
        df$Behaviour <- mapvalues(df$Behaviour, from=old_names, to=new_names)
        df$Behaviour <- mapvalues(df$Behaviour, from=old_names_conv, to=new_names)
    }
    if ("Regressor" %in% names(df)){
        df$Regressor <- mapvalues(df$Regressor, from=old_names, to=new_names)
        df$Regressor <- mapvalues(df$Regressor, from=old_names_conv, to=new_names)
    }
    return(df)
}

rename_flies <- function(df){
    old_names <- c("210830_1", "210910_2", "211026_3", "211027_4", "211029_5")
    new_names <- c("Fly 1", "Fly 2", "Fly 3", "Fly 4", "Fly 5")
    df$Fly <- mapvalues(df$Fly, from=old_names, to=new_names)
    return(df)
}

rename_angles <- function(df){
    old_names <- c(
           "Angle__LF_leg_yaw",    "Angle__LF_leg_pitch",  "Angle__LF_leg_roll",
           "Angle__LF_leg_th_fe",   "Angle__LF_leg_th_ti",   "Angle__LF_leg_roll_tr",
           "Angle__LF_leg_th_ta",   "Angle__LM_leg_yaw",     "Angle__LM_leg_pitch",
           "Angle__LM_leg_roll",    "Angle__LM_leg_th_fe",   "Angle__LM_leg_th_ti",
           "Angle__LM_leg_roll_tr", "Angle__LM_leg_th_ta",   "Angle__LH_leg_yaw",
           "Angle__LH_leg_pitch",   "Angle__LH_leg_roll",    "Angle__LH_leg_th_fe",
           "Angle__LH_leg_th_ti",   "Angle__LH_leg_roll_tr", "Angle__LH_leg_th_ta",
           "Angle__RF_leg_yaw",     "Angle__RF_leg_pitch",   "Angle__RF_leg_roll",
           "Angle__RF_leg_th_fe",   "Angle__RF_leg_th_ti",   "Angle__RF_leg_roll_tr",
           "Angle__RF_leg_th_ta",   "Angle__RM_leg_yaw",     "Angle__RM_leg_pitch",
           "Angle__RM_leg_roll",    "Angle__RM_leg_th_fe",   "Angle__RM_leg_th_ti",
           "Angle__RM_leg_roll_tr", "Angle__RM_leg_th_ta",   "Angle__RH_leg_yaw",
           "Angle__RH_leg_pitch",   "Angle__RH_leg_roll",    "Angle__RH_leg_th_fe",
           "Angle__RH_leg_th_ti",   "Angle__RH_leg_roll_tr", "Angle__RH_leg_th_ta"
          )
    old_names_conv <- paste(old_names, "conv", sep=".")
    new_names <- c(
           "LF ThC yaw",     "LF ThC pitch",   "LF ThC roll",
           "LF CTr pitch",   "LF FTi pitch",   "LF CTr roll",
           "LF TiTa pitch",   "LM ThC yaw",     "LM ThC pitch",
           "LM ThC roll",    "LM CTr pitch",   "LM FTi pitch",
           "LM CTr roll", "LM TiTa pitch",   "LH ThC yaw",
           "LH ThC pitch",   "LH ThC roll",    "LH CTr pitch",
           "LH FTi pitch",   "LH CTr roll", "LH TiTa pitch",
           "RF ThC yaw",     "RF ThC pitch",   "RF ThC roll",
           "RF CTr pitch",   "RF FTi pitch",   "RF CTr roll",
           "RF TiTa pitch",   "RM ThC yaw",     "RM ThC pitch",
           "RM ThC roll",    "RM CTr pitch",   "RM FTi pitch",
           "RM CTr roll", "RM TiTa pitch",   "RH ThC yaw",
           "RH ThC pitch",   "RH ThC roll",    "RH CTr pitch",
           "RH FTi pitch",   "RH CTr roll", "RH TiTa pitch"
          )
    for (i in 1:length(old_names)){
        names(df)[names(df) == old_names[i]] <- new_names[i]
        names(df)[names(df) == old_names_conv[i]] <- new_names[i]
    }
    if ("Regressor" %in% names(df)){
        df$Regressor <- mapvalues(df$Regressor, from=old_names, to=new_names)
        df$Regressor <- mapvalues(df$Regressor, from=old_names_conv, to=new_names)
    }
    return(df)
}

plot_matrix <- function(df, value_col, row.order, scale_name, output_file, f_p_value_df, x_label, range, text_col, as_perc, justifications, ROI_label_angle, faceting, y_axis_label, height, width){
    if (missing(x_label)){
        x_label <- ""
    }
    if (missing(range)){
        range <- c(min(df[, value_col]), max(df[, value_col]))
    }
    if (missing(ROI_label_angle)){
        ROI_label_angle <- 0
    }
    if (missing(height)){
        height <- 7
    }
    if (missing(width)){
        width <- 5
    }
    # Fix order of Fly_ROI
    df.row.order <- unique(match(row.order[["Fly_ROI"]], df$Fly_ROI))
    df.row.order <- df.row.order[!is.na(df.row.order)]
    df$Fly_ROI <- factor(df$Fly_ROI, levels = df$Fly_ROI[df.row.order])
    df <- merge(df, row.order, by=c("Fly", "ROI"))

    flies <- unique(df$Fly[df.row.order])
    fly_pos <- c()
    fly_line_start <- c()
    fly_line_stop <- c()
    for (fly in flies){
        fly_pos <- c(fly_pos, mean(which(df$Fly[df.row.order] %in% fly)))
        fly_line_start <- c(fly_line_start, min(which(df$Fly[df.row.order] %in% fly)))
        fly_line_stop <- c(fly_line_stop, max(which(df$Fly[df.row.order] %in% fly)))
    }
    flies <- lapply(flies, as.character)
    fly_colours <- replicate(length(flies), "black")
    fly_line_col <- replicate(length(flies), "black")
    if (length(flies) > 1){
        fly_line_col[seq(1, length(fly_line_col), 2)] <- "#646464"
    }

    if (!missing(f_p_value_df)){
        f_p_value_df.row.order <- unique(match(row.order, f_p_value_df$Fly_ROI))
        f_p_value_df.row.order <- f_p_value_df.row.order[!is.na(f_p_value_df.row.order)]
    }

    if (! value_col %in% names(df)){
        print(colnames(df))
        stop(paste(value_col, "is not a column of df"))
    }
    names(df)[names(df) == value_col] <- "value"
    if (missing(as_perc)){
        as_perc <- FALSE
    }
    if (as_perc){
        df$value <- df$value * 100
        range <- range * 100
    }

    n_rows <- nlevels(droplevels(as.factor(df$Regressor)))

    if (missing(justifications)){
        justifications <- c()
    }
    if (! "neuron_num_hjust" %in% names(justifications)){
        justifications <- c(justifications, neuron_num_hjust=1.1)
    }
    if (! "neuron_num_vjust" %in% names(justifications)){
        justifications <- c(justifications, neuron_num_vjust=-1.5/34*n_rows + 1000/34)
    }
    if (! "driver_line_hjust" %in% names(justifications)){
        justifications <- c(justifications, driver_line_hjust=1.1)
    }
    if (! "driver_line_vjust" %in% names(justifications)){
        justifications <- c(justifications, driver_line_vjust=5.8)
    }
    if (! "fly_hjust" %in% names(justifications)){
        justifications <- c(justifications, fly_hjust=2.3)
    }
    if (! "fly_vjust" %in% names(justifications)){
        justifications <- c(justifications, fly_vjust=0.51)
    }
    if (! "fly_line_y" %in% names(justifications)){
        justifications <- c(justifications, fly_line_y=-3.2/34 * n_rows + 15.4/34)
    }
    if (! "p_value_hjust" %in% names(justifications)){
        justifications <- c(justifications, p_value_hjust=1.1)
    }
    if (! "p_value_vjust" %in% names(justifications)){
        justifications <- c(justifications, p_value_vjust=0.8/34 * n_rows - 2.5)
    }
    if (! "star_hjust" %in% names(justifications)){
        justifications <- c(justifications, star_hjust=1/34 * n_rows + (-76/34))
    }
    if (! "star_vjust" %in% names(justifications)){
        justifications <- c(justifications, star_vjust=0.8)
    }

    x_labels <- rep("", length(as.vector(df$Fly_ROI.x)))
    names(x_labels) <- as.vector(df$New_ROI)


    levels <- c(
                "LH TiTa pitch", "RH TiTa pitch", "LH FTi pitch", "RH FTi pitch", "LH CTr roll", "RH CTr roll", "LH CTr pitch", "RH CTr pitch", "LH ThC roll", "RH ThC roll", "LH ThC pitch", "RH ThC pitch", "LH ThC yaw", "RH ThC yaw",
                "LM TiTa pitch", "RM TiTa pitch", "LM FTi pitch", "RM FTi pitch", "LM CTr roll", "RM CTr roll", "LM CTr pitch", "RM CTr pitch", "LM ThC roll", "RM ThC roll", "LM ThC pitch", "RM ThC pitch", "LM ThC yaw", "RM ThC yaw",
                "LF TiTa pitch", "RF TiTa pitch", "LF FTi pitch", "RF FTi pitch", "LF CTr roll", "RF CTr roll", "LF CTr pitch", "RF CTr pitch", "LF ThC roll", "RF ThC roll", "LF ThC pitch", "RF ThC pitch", "LF ThC yaw", "RF ThC yaw",
                "Roll", "Yaw", "Pitch",
                #"turn_l", "turn_r",
                "ACV", "MSC", "Behavior", "H2O", "all",
                "head_grooming", "all_anterior", "foreleg_grooming",
                "Proboscis extension", "Rear leg grooming", "Abdominal grooming", "Front leg grooming", "Antennal grooming", "Eye grooming", "Resting", "Backward walking", "Pushing", "Forward walking",
                "hind_angles", "middle_angles", "front_angles", "hind_r_angles", "hind_l_angles", "middle_r_angles", "middle_l_angles", "front_r_angles", "front_l_angles",
                "hind_positions", "middle_positions", "front_positions", "hind_r_positions", "hind_l_positions", "middle_r_positions", "middle_l_positions", "front_r_positions", "front_l_positions"
               )
    #levels <- c("turn_l", "turn_r", "Walking", "Posterior grooming", "Front leg rubbing", "Head grooming", "Resting")
    additional_levels <- as.vector(unlist(unique(df[!(df$Regressor %in% levels), "Regressor"])))
    df$Regressor <- factor(df$Regressor, levels=c(additional_levels, levels))
    #df$Regressor <- factor(df$Regressor, levels=levels)
    indices <- match(sort(unique(df$Regressor)), names(behaviour_colours))
    modified_behaviour_colours <- behaviour_colours
    y_tick_label_colours <- modified_behaviour_colours[indices]
    y_tick_label_colours <- replace_na(y_tick_label_colours, "black")
    size_factor <- 2.134

    if (faceting){
        p <- (ggplot(data=df, aes(x=New_ROI, y=Regressor, fill=value))
             + facet_grid(rows=vars(Fly))
             + scale_x_discrete(labels=x_labels)
             )
    } else {
        p <- ggplot(data=df, aes(x=Fly, y=Regressor, fill=value))
    }
    p <- p + geom_tile(colour="white", stat="identity") +
        scale_fill_gradient2(low=muted("red"), mid="white", high=muted("blue"), midpoint=max(0, range[1]), na.value="white", limits=range, space = "Lab") +
        #scale_fill_gradient2(low=muted("red"), mid="white", high=muted("red"), midpoint=max(0, range[1]), na.value="white", limits=range, space = "Lab") +
        theme(panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              #panel.background = element_blank(),
              panel.border = element_rect(colour="black", fill=NA, size=0.25 / size_factor),
              axis.text.x = element_text(family="Arial", colour="black", size=4, angle=ROI_label_angle),
              axis.text.y = element_text(family="Arial",
                                         colour=y_tick_label_colours,
                                         size=min(c(4, 42/n_rows))),
              axis.ticks = element_line(colour = "black", size=0.5 / size_factor),
              axis.ticks.length = unit(1.198, "pt"),
              text = element_text(family="Arial", colour = "black", size=6),
              legend.title=element_text(family="Arial", colour="black", size=4),
              legend.text=element_text(family="Arial", colour="black", size=4),
              legend.position="top",
              strip.text.y=element_text(family="Arial", colour="black", size=3, margin=margin(0.1, 0, 0.1, 0, "pt"))
              #strip.background = element_rect(color="black", size=1.5)
             ) +
        guides(fill = guide_colourbar(title.position="bottom", barwidth=unit(90, "pt"), frame.colour="black", frame.linewidth=0.25 / 0.76, direction="horizontal", barheight = unit(3, "pt"), ticks.colour="black", ticks.linewidth=0.5 / 0.76, label.position="top")) +
        coord_cartesian(expand = FALSE, clip = "off")
     if (faceting){
        p <- p + labs(x="ROI", y=y_axis_label, fill=scale_name)
     } else {
        p <- p + labs(x="", y=y_axis_label, fill=scale_name)
     }
    ggsave(output_file, height=height, width=width, scale=0.5, device=cairo_pdf, bg="transparent")
}
