# ======================================================================= #
# Time to pay attention?: Information search explains amplified framing
# effects under time pressure
# 
# Authors: Roberts, I.D., Teoh, Y.Y., & Hutcherson, C.A.
# ======================================================================= #

# setup -------------------------------------------------------------------
library(psych); library(tidyverse); library(lme4); library(lmerTest);
library(ez); library(influence.ME); library(ggpubr); library(effects);
library(effsize)

dat <- read_csv("ftp_osf_data.csv", col_types = cols())


# Time pressure X frame on choice -----------------------------------------

# P-H1 & P-H2 

# P-H1. Participants will be more likely to choose the ‘sure’ option when
#   a gain framing is presented versus a loss framing (i.e., the classical
#   framing effect).

# P-H2. There will an interaction between framing and time pressure such
#   that participants will show a greater difference in the proportion of
#   ‘sure’ options chosen between the gain and loss framings when time
#   pressure is applied versus when there is no time pressure (replicating
#   Guo et al.’s results).

# ANOVA
ph2_anova_dat <- dat %>% 
  filter(!is.na(choice) & trialType == "target") %>% 
  group_by(subject, cond, frame) %>% 
  summarize(propGamble = mean(choice)) %>% 
  ungroup() %>% 
  mutate(subject = factor(subject),
         cond = factor(cond),
         frame = factor(frame))

ph2_anova_mod <- ezANOVA(ph2_anova_dat, dv = .(propGamble), wid = .(subject),
                         within = .(cond, frame), type = 3, detailed = T)
ph2_anova_mod

# ANOVA simple effects
### Gain: NTC v TC
with(filter(ph2_anova_dat, frame == "gain"), t.test(propGamble ~ cond,
                                                    paired = T))
with(filter(ph2_anova_dat, frame == "gain"), effsize::cohen.d(propGamble, cond,
                                                              paired = T,
                                                              within = F,
                                                              subject = subject))
### Loss: NTC v TC
with(filter(ph2_anova_dat, frame == "loss"), t.test(propGamble ~ cond,
                                                    paired = T))
with(filter(ph2_anova_dat, frame == "loss"), effsize::cohen.d(propGamble, cond,
                                                              paired = T,
                                                              within = F,
                                                              subject = subject))

### NTC: Gain v Loss
with(filter(ph2_anova_dat, cond == "ntc"), t.test(propGamble ~ frame,
                                                  paired = T))
with(filter(ph2_anova_dat, cond == "ntc"), effsize::cohen.d(propGamble, frame,
                                                            paired = T,
                                                            within = F,
                                                            subject = subject))
### TC: Gain v Loss
with(filter(ph2_anova_dat, cond == "tc"), t.test(propGamble ~ frame,
                                                 paired = T))
with(filter(ph2_anova_dat, cond == "tc"), effsize::cohen.d(propGamble, frame,
                                                           paired = T,
                                                           within = F,
                                                           subject = subject))

# DESCRIPTIVES
### Cond X Frame: Mean & SD
ph2_anova_dat %>% group_by(frame, cond) %>%
  summarize(mean = round(mean(propGamble), 2),
            sd = round(sd(propGamble),2), .groups = "keep")

### Frame: Mean & SD
ph2_anova_dat %>% group_by(frame) %>%
  summarize(mean = round(mean(propGamble), 2),
            sd = round(sd(propGamble),2), .groups = "keep")

### Cond: Mean & SD
ph2_anova_dat %>% group_by(cond) %>%
  summarize(mean = round(mean(propGamble), 2),
            sd = round(sd(propGamble),2), .groups = "keep")


# P-H2 plot (gain and loss) -- Figure 2A
ph2_plotDat <- ph2_anova_dat %>% 
  mutate(cond = factor(cond, levels = c("ntc", "tc"),
                       labels = c("No Time Constraint",
                                  "Time Constraint")),
         frame = factor(frame, levels = c("gain", "loss"),
                        labels = c("Gain Frame", "Loss Frame")),
         framePos = as.numeric(frame),
         condPos = ifelse(cond == "No Time Constraint",
                          framePos-0.2, framePos+0.2)) %>% 
  group_by(subject) %>% 
  mutate(jitterPos = runif(1, min=-0.08, max=0.08) + condPos) %>% 
  unite(groupFactor, subject, frame, remove=F) %>% 
  ungroup()

plotMeans <- ph2_plotDat %>% 
  select(cond, frame, condPos, propGamble) %>% 
  group_by(cond, frame, condPos) %>% 
  summarize(meanPropGamble = mean(propGamble))

plotCI <- ph2_plotDat %>%
  group_by(subject) %>%
  mutate(propGamble = propGamble - mean(propGamble)) %>% 
  full_join(plotMeans[,c("cond", "frame", "meanPropGamble")],
            by = c("cond", "frame")) %>% 
  mutate(propGamble = propGamble + meanPropGamble) %>% 
  group_by(cond, frame) %>%
  summarize(ci = (qt(0.975, n()-1)*sd(propGamble)/sqrt(n())) * sqrt(4/(4-1)))

plotMeans <- full_join(plotMeans, plotCI, c("cond", "frame"))

fig2a <- ggplot(ph2_plotDat, aes(x=jitterPos, y=propGamble, color=cond,
                                 shape=cond)) +
  geom_line(aes(group=groupFactor), color="black", alpha=0.1) +
  geom_point(alpha=0.5, size = 2) +
  scale_x_continuous(breaks=c(1,2), labels=levels(ph2_plotDat$frame),
                     limits=c(0.5,2.5)) +
  scale_color_manual(values=c("orchid3", "chartreuse3")) +
  scale_shape_manual(values=c(16, 17)) +
  geom_errorbar(data=plotMeans, aes(x=condPos, y = meanPropGamble,
                                    ymax=meanPropGamble+ci,
                                    ymin=meanPropGamble-ci),
                color="black", size=1, width=0.1) +
  geom_errorbarh(data=plotMeans, aes(x=condPos, xmin=condPos-0.15,
                                     xmax=condPos+0.15, y=meanPropGamble),
                 color="black", size=1.5, height=0) +
  ylab("Probability of Choosing Gamble") +
  theme_classic() +
  ylim(0,1) +
  theme(plot.title = element_text(hjust = 0.5, size = 20),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        legend.title = element_blank(),
        legend.position = "top",
        legend.text = element_text(size = 8),
        legend.box.background = element_rect(color = "black", size = 1),
        strip.background = element_rect(color = NA),
        strip.text = element_text(size = 14)) +
  guides(color = guide_legend(override.aes = list(size=5)))




# Effects of frame X time constraints on early attention ------------------

# P-H3a 

# P-H3. Participants will allocate more attention to the ‘sure’ option
#   under time pressure versus no time pressure.
# a. Participants will be more likely to fixate on the ‘sure’ option first
#   when there is time pressure versus no time pressure.

# ANOVA
ph3a_anova_dat <- dat %>% 
  filter(!is.na(firstFix) & trialType == "target") %>% 
  select(subject, trial, cond, frame, firstFix) %>% 
  unique() %>% 
  mutate(fixGambleFirst = case_when(firstFix == "gamble" ~ 1,
                                    firstFix == "sure" ~ 0,
                                    TRUE ~ NA_real_)) %>% 
  group_by(subject, cond, frame) %>% 
  summarize(propGambleFirst = sum(fixGambleFirst) / n()) %>% 
  ungroup() %>% 
  mutate(subject = factor(subject),
         cond = factor(cond),
         frame = factor(frame))

ph3a_anova_mod <- ezANOVA(ph3a_anova_dat, dv = .(propGambleFirst),
                          wid = .(subject), within = .(cond, frame), type = 3,
                          detailed = T)
ph3a_anova_mod


# ANOVA simple effects
### Gain: NTC v TC
with(filter(ph3a_anova_dat, frame == "gain"), t.test(propGambleFirst ~ cond,
                                                     paired = T))
with(filter(ph3a_anova_dat, frame == "gain"),
     effsize::cohen.d(propGambleFirst, cond, paired = T, within = F,
                      subject = subject))

### Loss: NTC v TC
with(filter(ph3a_anova_dat, frame == "loss"), t.test(propGambleFirst ~ cond,
                                                     paired = T))
with(filter(ph3a_anova_dat, frame == "loss"),
     effsize::cohen.d(propGambleFirst, cond, paired = T, within = F,
                      subject = subject))

### NTC: Gain v Loss
with(filter(ph3a_anova_dat, cond == "ntc"), t.test(propGambleFirst ~ frame,
                                                   paired = T))
with(filter(ph3a_anova_dat, cond == "ntc"),
     effsize::cohen.d(propGambleFirst, frame, paired = T, within = F,
                      subject = subject))

### TC: Gain v Loss
with(filter(ph3a_anova_dat, cond == "tc"), t.test(propGambleFirst ~ frame,
                                                  paired = T))
with(filter(ph3a_anova_dat, cond == "tc"),
     effsize::cohen.d(propGambleFirst, frame, paired = T, within = F,
                      subject = subject))

# DESCRIPTIVES
### Cond X Frame: Mean & SD
ph3a_anova_dat %>% group_by(frame, cond) %>%
  summarize(mean = round(mean(propGambleFirst), 2),
            sd = round(sd(propGambleFirst),2), .groups = "keep")

### Frame: Mean & SD
ph3a_anova_dat %>% group_by(frame) %>%
  summarize(mean = round(mean(propGambleFirst), 2),
            sd = round(sd(propGambleFirst),2), .groups = "keep")

### Cond: Mean & SD
ph3a_anova_dat %>% group_by(cond) %>%
  summarize(mean = round(mean(propGambleFirst), 2),
            sd = round(sd(propGambleFirst),2), .groups = "keep")


# P-H3a plot (gain and loss)
ph3a_plotDat <- ph3a_anova_dat %>% 
  mutate(cond = factor(cond, levels = c("ntc", "tc"),
                       labels = c("No Time Constraint",
                                  "Time Constraint")),
         frame = factor(frame, levels = c("gain", "loss"),
                        labels = c("Gain Frame", "Loss Frame")),
         framePos = as.numeric(frame),
         condPos = ifelse(cond == "No Time Constraint",
                          framePos-0.2, framePos+0.2)) %>% 
  group_by(subject) %>% 
  mutate(jitterPos = runif(1, min=-0.08, max=0.08) + condPos) %>% 
  unite(groupFactor, subject, frame, remove=F) %>% 
  ungroup()

plotMeans <- ph3a_plotDat %>% 
  select(cond, frame, condPos, propGambleFirst) %>% 
  group_by(cond, frame, condPos) %>% 
  summarize(meanPropGambleFirst = mean(propGambleFirst))

plotCI <- ph3a_plotDat %>%
  group_by(subject) %>%
  mutate(propGambleFirst = propGambleFirst - mean(propGambleFirst)) %>% 
  full_join(plotMeans[,c("cond", "frame", "meanPropGambleFirst")],
            by = c("cond", "frame")) %>% 
  mutate(propGambleFirst = propGambleFirst + meanPropGambleFirst) %>% 
  group_by(cond, frame) %>%
  summarize(ci = (qt(0.975, n()-1)*sd(propGambleFirst)/sqrt(n())) * sqrt(4/(4-1)))

plotMeans <- full_join(plotMeans, plotCI, c("cond", "frame"))

fig2b <- ggplot(ph3a_plotDat, aes(x=jitterPos, y=propGambleFirst, color=cond,
                                  shape=cond)) +
  geom_line(aes(group=groupFactor), color="black", alpha=0.1) +
  geom_point(alpha=0.5, size=2) +
  scale_x_continuous(breaks=c(1,2), labels=levels(ph3a_plotDat$frame),
                     limits=c(0.5,2.5)) +
  scale_color_manual(values=c("orchid3", "chartreuse3")) +
  scale_shape_manual(values=c(16, 17)) +
  geom_errorbar(data=plotMeans, aes(x=condPos, y = meanPropGambleFirst,
                                    ymax=meanPropGambleFirst+ci,
                                    ymin=meanPropGambleFirst-ci),
                color="black", size=1, width=0.1) +
  geom_errorbarh(data=plotMeans, aes(x=condPos, xmin=condPos-0.15,
                                     xmax=condPos+0.15, y=meanPropGambleFirst),
                 color="black", size=1.5, height=0) +
  ylab("Probability Fixate Gamble First") +
  theme_classic() +
  ylim(0,1) +
  theme(plot.title = element_text(hjust = 0.5, size = 20),
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        legend.title = element_blank(),
        legend.position = "top",
        legend.text = element_text(size = 8),
        legend.box.background = element_rect(color = "black", size = 1),
        strip.background = element_rect(color = NA),
        strip.text = element_text(size = 14)) +
  guides(color = guide_legend(override.aes = list(size=5)))


fig2 <- ggarrange(fig2a +
                    theme(legend.position = c(0.25, 0.95),
                          legend.background = element_rect(fill = "transparent",
                                                           color = NA),
                          legend.box.background = element_rect(fill = "transparent",
                                                               color = NA),
                          strip.background = element_rect(color = NA),
                          legend.text = element_text(size = 10)),
                  fig2b + theme(legend.position = "none"),
                  labels = c("A", "B"))



# Effects of peripheral vision on early attention -------------------------

# T-H3g

# T-H3. Moderation of the effect of time pressure on attention allocation
#   (P-H3)
# g. The effect of time pressure on attention allocation may be reduced
#   when the gamble is a more extreme probability (e.g., because more
#   extreme probabilities may make it more difficult to initially
#   determine which option is the ‘sure’ option with peripheral vision).

th3g_glmer_dat <- dat %>% 
  mutate(cond = factor(cond),
         frame = factor(frame),
         fixGambleFirst = case_when(firstFix == "gamble" ~ 1,
                                    firstFix == "sure" ~ 0)) %>% 
  filter(!is.na(fixGambleFirst)) %>%
  select(subject, trial, fixGambleFirst, prob, cond, frame, endow) %>% 
  unique()
contrasts(th3g_glmer_dat$cond) <- contr.treatment(levels(th3g_glmer_dat$cond),
                                                  base = 1)
contrasts(th3g_glmer_dat$frame) <- contr.sum(levels(th3g_glmer_dat$frame))

th3g_glmer_prob_mod <- glmer(fixGambleFirst ~ cond*prob + cond*frame +
                               (1 + cond + frame + prob | subject),
                             data = th3g_glmer_dat, family = "binomial",
                             control = glmerControl(optimizer = "bobyqa",
                                                    optCtrl = list(maxfun = 200000)))
summary(th3g_glmer_prob_mod)


# T-H3g plot
modEffs <- Effect(c("prob", "cond", "frame"), th3g_glmer_prob_mod,
                  xlevels = list(prob = seq(min(th3g_glmer_dat$prob, na.rm = T),
                                            max(th3g_glmer_dat$prob, na.rm = T),
                                            0.01)))

modFit <- as_tibble(summary(modEffs)$effect) %>% 
  mutate(prob = as.numeric(rownames(summary(modEffs)$effect))) %>% 
  gather(cond, fixGambleFirst, -prob)

modLowerCI <- as_tibble(summary(modEffs)$lower) %>% 
  mutate(prob = as.numeric(rownames(summary(modEffs)$effect))) %>% 
  gather(cond, lowerCI, -prob)

modUpperCI <- as_tibble(summary(modEffs)$upper) %>% 
  mutate(prob = as.numeric(rownames(summary(modEffs)$effect))) %>% 
  gather(cond, upperCI, -prob)

plotDat <- full_join(modFit, modLowerCI, by = c("cond", "prob")) %>% 
  full_join(modUpperCI, by = c("cond", "prob")) %>% 
  separate(cond, c("cond", "frame")) %>%
  mutate(cond = factor(cond, levels = c("ntc", "tc"),
                       labels = c("No Time Constraint", "Time Constraint")),
         frame = factor(frame, levels = c("gain", "loss"),
                        labels = c("Gain Frame", "Loss Frame")))

plotDat_points <- th3g_glmer_dat %>% 
  mutate(prob = round(prob, 2)) %>% 
  group_by(cond, frame, prob) %>% 
  summarize(probFixGambleFirst = mean(fixGambleFirst)) %>% 
  ungroup() %>% 
  mutate(cond = factor(cond, levels = c("ntc", "tc"),
                       labels = c("No Time Constraint", "Time Constraint")),
         frame = factor(frame, levels = c("gain", "loss"),
                        labels = c("Gain Frame", "Loss Frame")))

large_points <- function(data, params, size) {
  # Multiply by some number
  data$size <- data$size * 2.5
  draw_key_point(data = data, params = params, size = size)
}

small_line <- function(data, params, size) {
  # Multiply by some number
  data$size <- data$size / 2
  draw_key_smooth(data = data, params = params, size = size)
}

fig3 <- ggplot(plotDat, aes(x = prob,
                            y = fixGambleFirst,
                            color = cond, fill = cond, shape = cond,
                            linetype = cond)) +
  geom_point(data = plotDat_points, aes(x = prob, y = probFixGambleFirst,
                                        color = cond),
             fill = NA, alpha = 0.3, size = 2, key_glyph = large_points) +
  geom_ribbon(data = filter(plotDat, cond == "Time Constraint"),
              aes(x = prob, ymin = lowerCI, ymax = upperCI,
                  fill = "Time Constraint"), alpha = 0.3, color = NA) +
  geom_line(data = filter(plotDat, cond == "Time Constraint"),
            aes(x = prob, y = fixGambleFirst, color = "Time Constraint"),
            size = 2, key_glyph = small_line) +
  geom_ribbon(data = filter(plotDat, cond == "No Time Constraint"),
              aes(x = prob, ymin = lowerCI, ymax = upperCI,
                  fill = "No Time Constraint"), alpha = 0.3, color = NA) +
  geom_line(data = filter(plotDat, cond == "No Time Constraint"),
            aes(x = prob, y = fixGambleFirst, color = "No Time Constraint"),
            size = 2, key_glyph = small_line) +
  ylim(0,1) +
  scale_color_manual(values = c("Time Constraint" = "chartreuse3",
                                "No Time Constraint" = "orchid3")) +
  scale_shape_manual(values = c("Time Constraint" = 17,
                                "No Time Constraint" = 16)) +
  scale_linetype_manual(values = c("Time Constraint" = 2,
                                   "No Time Constraint" = 1)) +
  scale_fill_manual(values = c("Time Constraint" = "chartreuse3",
                               "No Time Constraint" = "orchid3"),
                    guide = "none") +
  ylab("Probability of Fixating Gamble First") +
  xlab("Probability of Winning Gamble") +
  facet_grid(. ~ frame) +
  theme_classic() +
  theme(legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        legend.text = element_text(size = 10),
        strip.background = element_rect(color = NA),
        strip.text = element_text(size = 14),
        legend.position = c(0.15, 0.9),
        aspect.ratio = 1) +
  guides(linetype = guide_legend(override.aes = list(fill=NA)))



# Effects of attention X time constraints on choice -----------------------

# P-H4a

# P-H4. Allocating attention to the ‘sure’ option under time pressure will
#   increase magnitude of the framing effect.
# a. Time pressure trials on which a participant fixates on the ‘sure’ option
#   first will show an increased likelihood of a framing-effect-consistent
#   choice.


# mixed-effects logistic regression
ph4a_glmer_dat <- dat %>% 
  filter(trialType == "target") %>%
  ungroup() %>% 
  mutate(cond = factor(cond),
         frame = factor(frame),
         optsFix = factor(fixBoth, levels = c(0, 1), labels = c("one", "both")),
         sureSide = factor(sureSide),
         firstFix = factor(firstFix)) %>%
  select(subject, trial, cond, frame, firstFix, sureSide, choice, fixBoth,
         optsFix) %>%
  unique()

contrasts(ph4a_glmer_dat$cond) <- contr.treatment(levels(ph4a_glmer_dat$cond),
                                                  base = 1)
contrasts(ph4a_glmer_dat$frame) <- contr.treatment(levels(ph4a_glmer_dat$frame),
                                                   base = 1)
contrasts(ph4a_glmer_dat$firstFix) <- contr.treatment(levels(ph4a_glmer_dat$firstFix),
                                                      base = 2)
contrasts(ph4a_glmer_dat$optsFix) <- contr.treatment(levels(ph4a_glmer_dat$optsFix),
                                                     base = 1)

ph4a_glmer_frame_mod <- glmer(choice ~ cond*frame*firstFix +
                                (1 + cond + frame + firstFix | subject),
                              data = ph4a_glmer_dat, family="binomial",
                              control = glmerControl(optimizer = "bobyqa",
                                                     optCtrl = list(maxfun = 100000)))
summary(ph4a_glmer_frame_mod)


# gate-keeping vs primacy effects of attention
ph4a_glmer_optsFix_mod <- glmer(choice ~ cond*firstFix*optsFix +
                                  cond*frame + frame*firstFix + optsFix*frame +
                                  (1 + cond + frame + firstFix + optsFix | subject),
                                data = ph4a_glmer_dat, family="binomial",
                                control = glmerControl(optimizer = "bobyqa",
                                                       optCtrl = list(maxfun = 100000)))
summary(ph4a_glmer_optsFix_mod)

# descriptives: frequency of fixating one or both options
ph4a_glmer_dat %>% 
  filter(!is.na(firstFix)) %>% 
  group_by(cond, optsFix) %>% 
  summarize(N = n()) %>% 
  group_by(cond) %>% 
  mutate(prop = N / sum(N))


# full plot -- time constraint on y facet
plotDat_points <- ph4a_glmer_dat %>% 
  filter(!is.na(firstFix)) %>% 
  group_by(subject, cond, frame, firstFix, optsFix) %>% 
  summarize(probGamb = mean(choice, na.rm = T),
            N = n(), .groups = "keep") %>% 
  mutate(cond = factor(cond, levels = c("ntc", "tc"),
                       labels = c("No Time\nConstraint", "Time\nConstraint")),
         frame = factor(frame, levels = c("gain", "loss"),
                        labels = c("Gain Frame", "Loss Frame")),
         optsFix = factor(optsFix, levels = c("one", "both"),
                          labels = c("One", "Both")),
         firstFix = factor(firstFix, levels = c("sure", "gamble"),
                           labels = c("Sure", "Gamble")),
         nOptsPos = as.numeric(optsFix),
         fixPos = ifelse(firstFix == "Sure", nOptsPos-0.2, nOptsPos+0.2)) %>% 
  group_by(subject) %>% 
  mutate(jitterPos = runif(1, min=-0.08, max=0.08) + fixPos) %>% 
  unite(groupFactor, subject, optsFix, remove=F) %>% 
  ungroup()

plotMeans <- plotDat_points %>% 
  select(cond, frame, firstFix, optsFix, fixPos, probGamb) %>% 
  group_by(cond, frame, firstFix, optsFix, fixPos) %>% 
  summarize(meanProbGamb = mean(probGamb, na.rm = T),
            N = n(), .groups = "keep")

plotCI <- plotDat_points %>%
  group_by(subject) %>%
  mutate(probGamb = probGamb - mean(probGamb, na.rm = T)) %>% 
  full_join(plotMeans[,c("cond", "frame", "firstFix", "optsFix",
                         "meanProbGamb")], by=c("cond", "frame", "optsFix",
                                                "firstFix")) %>% 
  mutate(probGamb = probGamb + meanProbGamb) %>% 
  group_by(cond, frame, firstFix, optsFix) %>%
  summarize(ci = (qt(0.975, n()-1)*sd(probGamb)/sqrt(n())) * sqrt(16/(16-1)),
            .groups = "keep")

plotMeans <- full_join(plotMeans, plotCI, c("cond", "frame", "firstFix",
                                            "optsFix"))

fig4 <- ggplot(plotDat_points, aes(x=jitterPos, y=probGamb, color=firstFix,
                                   shape=firstFix)) +
  geom_hline(yintercept = 0.5, linetype = 2, alpha = 0.3) +
  geom_line(aes(group=groupFactor), color="black", alpha=0.1) +
  geom_point(alpha=0.4, size=2) +
  scale_x_continuous(breaks=c(1,2), labels=levels(plotDat_points$optsFix),
                     limits=c(0.5,2.5)) +
  coord_cartesian(ylim = c(0,1), clip = "off") +
  scale_color_manual("First Fixation",
                     values = c("Sure" = "deepskyblue",
                                "Gamble" = "tomato")) +
  scale_shape_manual("First Fixation",
                     values = c("Sure" = 16,
                                "Gamble" = 17)) +
  geom_errorbar(data=plotMeans, aes(x=fixPos, ymax=meanProbGamb+ci,
                                    ymin=meanProbGamb-ci, y=meanProbGamb),
                color="black", size=1, width=0.1) +
  geom_errorbarh(data=plotMeans, aes(x=fixPos, xmin=fixPos-0.15,
                                     xmax=fixPos+0.15, y=meanProbGamb),
                 color="black", size=1.5, height=0) +
  ylab("Probability of Choosing Gamble") +
  xlab("Number of Options Fixated") +
  facet_grid(cond ~ frame) +
  theme_classic() +
  theme(axis.title.x = element_text(size = 12),
        axis.text.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        legend.title = element_text(size = 12),
        legend.position = "top", #c(0.1, 0.9),
        strip.background = element_rect(color = NA),
        strip.text = element_text(size = 14)) +
  guides(color = guide_legend(title.position = "top", title.hjust = 0.6,
                              override.aes = list(size=5)))



# Effects of time constraint and expected value on information sea --------

# Post-hoc analysis: Are participants more likely to fixate the other
#   option under time pressure depending on the value of the first fixated
#   option?

infoSearch_dat <- dat %>% 
  filter(trialType == "target" & !is.na(firstFix)) %>%
  mutate(cond = factor(cond),
         frame = factor(frame),
         firstFix = factor(firstFix),
         subject = factor(subject),
         sureEV = ifelse(frame == "gain", sureOutcome, endow+sureOutcome)) %>% 
  select(subject, trial, cond, frame, firstFix, fixBoth, sureEV, endow, prob) %>% 
  unique()

# re-scale and mean-center sure EV
sureEV_normDiv <- 100
infoSearch_dat$sureEV_norm <- infoSearch_dat$sureEV / sureEV_normDiv
sureEV_norm_mean <- mean(infoSearch_dat$sureEV_norm)
infoSearch_dat$sureEV_norm_cen <- infoSearch_dat$sureEV_norm - sureEV_norm_mean

contrasts(infoSearch_dat$cond) <- contr.treatment(levels(infoSearch_dat$cond),
                                                  base = 1)
contrasts(infoSearch_dat$frame) <- contr.treatment(levels(infoSearch_dat$frame),
                                                   base = 1)
contrasts(infoSearch_dat$firstFix) <- contr.treatment(levels(infoSearch_dat$firstFix),
                                                      base = 2)

# filter to just time constraint condition
infoSearch_glmer_tc_ev_mod <- glmer(fixBoth ~ sureEV_norm_cen*frame*firstFix +
                                      (1 + frame + firstFix | subject),
                                    data = filter(infoSearch_dat, cond == "tc"),
                                    family="binomial",
                                    control = glmerControl(optimizer = "bobyqa",
                                                           optCtrl = list(maxfun = 100000)))
summary(infoSearch_glmer_tc_ev_mod)


# plot
modEffs <- Effect(c("sureEV_norm_cen", "frame", "firstFix"),
                  infoSearch_glmer_tc_ev_mod,
                  xlevels = list(sureEV_norm_cen = seq(min(infoSearch_dat$sureEV_norm_cen, na.rm = T),
                                                       max(infoSearch_dat$sureEV_norm_cen, na.rm = T),
                                                       0.01)))

modFit <- as_tibble(summary(modEffs)$effect) %>% 
  mutate(sureEV_norm_cen = as.numeric(rownames(summary(modEffs)$effect))) %>% 
  gather(frame, probFixOther, -sureEV_norm_cen)

modLowerCI <- as_tibble(summary(modEffs)$lower) %>% 
  mutate(sureEV_norm_cen = as.numeric(rownames(summary(modEffs)$effect))) %>% 
  gather(frame, lowerCI, -sureEV_norm_cen)

modUpperCI <- as_tibble(summary(modEffs)$upper) %>% 
  mutate(sureEV_norm_cen = as.numeric(rownames(summary(modEffs)$effect))) %>% 
  gather(frame, upperCI, -sureEV_norm_cen)

plotDat <- full_join(modFit, modLowerCI, by = c("frame", "sureEV_norm_cen")) %>% 
  full_join(modUpperCI, by = c("frame", "sureEV_norm_cen")) %>% 
  separate(frame, c("frame", "firstFix")) %>%
  mutate(frame = factor(frame, levels = c("gain", "loss"),
                        labels = c("Gain Frame", "Loss Frame")),
         firstFix = factor(firstFix, levels = c("sure", "gamble"),
                           labels = c("Sure", "Gamble")),
         sureEV_norm = sureEV_norm_cen + sureEV_norm_mean,
         sureEV = sureEV_norm * sureEV_normDiv) %>% 
  filter(!is.na(firstFix))

plotDat_points <- infoSearch_dat %>% 
  filter(cond == "tc") %>% 
  mutate(sureEV = round(sureEV, 2)) %>% 
  group_by(frame, firstFix, sureEV) %>% 
  summarize(probFixOther = mean(fixBoth)) %>% 
  ungroup() %>% 
  mutate(frame = factor(frame, levels = c("gain", "loss"),
                        labels = c("Gain Frame", "Loss Frame")),
         firstFix = factor(firstFix, levels = c("sure", "gamble"),
                           labels = c("Sure", "Gamble"))) %>% 
  filter(!is.na(firstFix))

large_points <- function(data, params, size) {
  # Multiply by some number
  data$size <- data$size * 2.5
  draw_key_point(data = data, params = params, size = size)
}

small_line <- function(data, params, size) {
  # Multiply by some number
  data$size <- data$size / 2
  draw_key_smooth(data = data, params = params, size = size)
}

fig5 <- ggplot(plotDat, aes(x = sureEV,
                            y = probFixOther,
                            color = firstFix, fill = firstFix,
                            shape = firstFix, linetype = firstFix)) +
  geom_point(data = plotDat_points, aes(x = sureEV, y = probFixOther,
                                        color = firstFix),
             fill = NA, alpha = 0.3, size = 2, key_glyph = large_points) +
  geom_ribbon(data = filter(plotDat, firstFix == "Gamble"),
              aes(x = sureEV, ymin = lowerCI, ymax = upperCI,
                  fill = "Gamble"), alpha = 0.3, color = NA) +
  geom_line(data = filter(plotDat, firstFix == "Gamble"),
            aes(x = sureEV, y = probFixOther, color = "Gamble"),
            size = 2, key_glyph = small_line) +
  geom_ribbon(data = filter(plotDat, firstFix == "Sure"),
              aes(x = sureEV, ymin = lowerCI, ymax = upperCI,
                  fill = "Sure"), alpha = 0.3, color = NA) +
  geom_line(data = filter(plotDat, firstFix == "Sure"),
            aes(x = sureEV, y = probFixOther, color = "Sure"),
            size = 2, key_glyph = small_line) +
  ylim(0,1) +
  scale_color_manual(values = c("Sure" = "deepskyblue",
                                "Gamble" = "tomato")) +
  scale_linetype_manual(values = c("Sure" = 1,
                                   "Gamble" = 2)) +
  scale_shape_manual(values = c("Sure" = 16,
                                "Gamble" = 17)) +
  scale_fill_manual(values = c("Sure" = "deepskyblue",
                               "Gamble" = "tomato"),
                    guide = "none") +
  ylab("Probability of Fixating Other Option") +
  xlab("Expected Value of First Fixated Option") +
  facet_grid(. ~ frame) +
  theme_classic() +
  theme(legend.title = element_blank(),
        legend.background = element_rect(fill = "transparent"),
        legend.text = element_text(size = 10),
        strip.background = element_rect(color = NA),
        strip.text = element_text(size = 14),
        legend.position = c(0.1, 0.9),
        aspect.ratio = 1) +
  guides(color = guide_legend(override.aes = list(fill=NA)))
