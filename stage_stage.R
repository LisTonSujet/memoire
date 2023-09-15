library(tidyverse)#manipulation des dataset
#library(ggplot2)#graphiques (compris dans tidyverse)
library(readxl)#importation de doc .xls
library(writexl)
library(ggplot2)
library(stats)
library(rstatix)
library(multcomp)
library(car)
library(ggpubr)
library(rcompanion)
library(emmeans)
library(ggsignif)
library(agricolae)
library(pals)
library(MASS)
library(PMCMR)
library(PMCMRplus)

##importation

df<- read_excel(file.choose(),col_names = T)
df<-as.data.frame(df)
df$median_DGCI <- as.numeric(df$median_DGCI)
df<- df %>%
  mutate(log_median = log10(df$median_DGCI))

#mise en facteurs 
df$association <- as.factor(df$association)
df$appareil <- as.factor(df$appareil)
df$stade_developpement <- as.factor(df$stade_developpement)
df$IA <- as.factor(df$IA)
str(df)

df$asso_triticale <- ifelse(df$association == "TTH_0N", "triticale_seul",
                                 ifelse(df$association %in% c("Mixte", "Rang"), "triticale_associé", NA))
df$asso_triticale <- as.factor(df$asso_triticale)
attach(df)

bxp_trit <- ggplot(df, aes(x=asso_triticale, y=log_median, colour=asso_triticale))+
  geom_boxplot(outlier.alpha = 0, alpha=0.25)+
  geom_jitter(width=0.25)+  
  stat_summary(fun=mean, colour="black", geom="point", shape=18, size=3) +
  theme_classic()+
  theme(legend.position="none")+
  theme(axis.text.x = element_text(angle=30, hjust=1, vjust=1))+
  labs(title="Log_median du DGCI avec présence de la féverole ou non", x= asso_triticale, y = "log_median")+
  theme(plot.title = element_text(hjust = 0.5))
stat.test <- df %>%
  t_test(log_median ~ asso_triticale) %>%
  add_significance()
stat.test
stat.test <- stat.test %>% add_xy_position(x = "asso_triticale")
bxp_trit + stat_pvalue_manual(stat.test, label = "p.signif", tip.length = 0.01)

#modélisation
lm1 <- lm(df$log_median~association*appareil*stade_developpement*IA, data=df) 
shapiro.test(residuals(lm1))#normalitÃ© des résidus ok
bartlett.test(residuals(lm1)~association)
qqnorm(residuals(lm1))
qqline(residuals(lm1)) 


#anova à 4 facteurs
df3 <- df %>%
  group_by(association, appareil, stade_developpement,IA) %>%
  get_summary_stats(log_median, type = "mean_sd")
modelcomplet  <- lm(log_median ~ association*appareil*stade_developpement*IA, data = df)
# Créer un QQ plot des résidus
ggqqplot(residuals(modelcomplet),
         xlab = "Quantiles",
         ylab = "Résidues observés",
         title ="Diagramme quantile-quantile \n du modèle d'interaction à 4 facteurs")+
  theme(plot.title = element_text(hjust = 0.5))
# Calculer le test de normalité de Shapiro-Wilk
shapiro_test(residuals(modelcomplet))
#grille de QQplot
ggqqplot(df, "log_median", ggtheme = theme_bw(), xlab= "quantile", y= "échantillons", title = "Distribution des quantiles \n pour chaque combinaison de facteurs") +
  facet_grid(appareil ~ association + stade_developpement + IA)+
  theme(plot.title = element_text(hjust = 0.5))

res.aov <-aov(log_median ~ association*appareil*stade_developpement*IA, data = df)
res.aov#interaction a 4 facteurs
summary(res.aov)

#création des facteurs combinés
df <- df %>% mutate(asso_stade_IA = paste(df$association, df$stade_developpement, df$IA))
df <- df %>% mutate(asso_stade = paste(df$association, df$stade_developpement))
df <- df %>% mutate(asso_IA = paste(df$association, df$IA))
df <- df %>% mutate(IA_stade = paste(df$IA, df$stade_developpement))
df$asso_stade_IA <- as.factor(df$asso_stade_IA)
df$asso_stade <- as.factor(df$app_stade)
df$asso_IA <- as.factor(df$asso_IA)
df$IA_stade <- as.factor(df$IA_stade)


#ANOVA 2 facteurs et boxplot asso-IA

aov_asso_IA <-aov(log_median ~ asso_IA*stade_developpement, data = df)
aov_asso_IA#interaction a 3 facteurs
summary(aov_asso_IA)
df$asso_IA <- factor(df$asso_IA, levels = c("TTH_0N YOLOv5", "TTH_0N UNet", "Rang YOLOv5", "Rang UNet", "Mixte YOLOv5", "Mixte UNet"))
lm_asso_IA <- lm(log_median ~ asso_IA*stade_developpement, data = df)
stat.test <- df %>%
  group_by(asso_IA) %>%
  t_test(log_median ~ stade_developpement) %>%
  adjust_pvalue() %>%
  add_significance("p.adj")
stat.test
bxp <- ggboxplot(
  df, x = "asso_IA", y = "log_median",
  color = "stade_developpement", palette = "jco",facet.by = "asso_IA"
)+
  labs(title = " log_median DGCI en fonction \n du facteur combiné association-IA \n et du stade de développement",y= "log_median DGCI")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
bxp
bxp <- bxp + facet_wrap(~ asso_IA, ncol=2)
stat.test <- stat.test %>% add_xy_position(x = "asso_IA")
bxp + 
  stat_pvalue_manual(stat.test, label = "p.adj.signif") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.10)))

#ANOVA 2 facteurs et boxplot IA-stade_developpement

aov_IA_stade <-aov(log_median ~ IA_stade*association, data = df)
aov_IA_stade#interaction a 3 facteurs
summary(aov_asso_IA)
lm_IA_stade <- lm(log_median ~ IA_stade*association, data = df)
stat.test <- df %>%
  group_by(IA_stade) %>%
  t_test(log_median ~ association) %>%
  adjust_pvalue() %>%
  add_significance("p.adj")
stat.test
bxp <- ggboxplot(
  df, x = "IA_stade", y = "log_median",
  color = "association", palette = "jco",facet.by = "IA_stade"
)+
  labs(title = " log_median DGCI en fonction \n du facteur combiné IA-stade de développement \n et de l'association",y= "log_median DGCI")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
bxp
bxp <- bxp + facet_wrap(~ IA_stade, ncol=2)
stat.test <- stat.test %>% add_xy_position(x = "IA_stade")
bxp + 
  stat_pvalue_manual(stat.test, label = "p.adj.signif") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.10)))


#ANOVA 2 facteurs et boxplot asso-stade

aov_asso_stade <-aov(log_median ~ asso_stade*IA, data = df)
aov_asso_stade#interaction a 3 facteurs
summary(aov_asso_stade)
df$asso_stade <- factor(df$asso_stade, levels = c("TTH_0N 1Noeud", "TTH_0N 2Noeud", "Rang 1Noeud", "Rang 2Noeud", "Mixte 1Noeud", "Mixte 2Noeud"))
lm_asso_stade <- lm(log_median ~ asso_stade*IA, data = df)
stat.test <- df %>%
  group_by(asso_stade) %>%
  t_test(log_median ~ IA) %>%
  adjust_pvalue() %>%
  add_significance("p.adj")
stat.test
bxp <- ggboxplot(
  df, x = "asso_stade", y = "log_median",
  color = "IA", palette = "jco",facet.by = "asso_stade"
)+
  labs(title = " log_median DGCI en fonction \n du facteur combiné association-stade de développement \n et de l'IA utilisée",y= "log_median DGCI")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
bxp
bxp <- bxp + facet_wrap(~ asso_stade, ncol=2)
stat.test <- stat.test %>% add_xy_position(x = "asso_stade")
bxp + 
  stat_pvalue_manual(stat.test, label = "p.adj.signif") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.10)))


#influence appareil
aov_asso_stade_IA <-aov(log_median ~ asso_stade_IA*appareil, data = df)
aov_asso_stade_IA#interaction a 3 facteurs
summary(aov_asso_stade_IA)
df$asso_stade_IA <- factor(df$asso_stade_IA, levels = c("Mixte 1Noeud UNet", "Rang 1Noeud UNet", "TTH_0N 1Noeud UNet",
                                             "Mixte 1Noeud YOLOv5","Rang 1Noeud YOLOv5","TTH_0N 1Noeud YOLOv5",
                                             "Mixte 2Noeud UNet","Rang 2Noeud UNet","TTH_0N 2Noeud UNet",
                                             "Mixte 2Noeud YOLOv5", "Rang 2Noeud YOLOv5","TTH_0N 2Noeud YOLOv5"))
lm_asso_stade_IA <- lm(log_median ~ asso_stade_IA, data = df)
stat.test <- df %>%
  group_by(asso_stade_IA) %>%
  t_test(log_median ~ appareil) %>%
  adjust_pvalue() %>%
  add_significance("p.adj")
stat.test
bxp <- ggboxplot(
  df, x = "asso_stade_IA", y = "log_median",
  color = "appareil", palette = "jco",facet.by = "asso_stade_IA"
)+
  labs(title = " log_median DGCI en fonction \n du facteur combiné association-stade-IA de développement \n et de l'appareil utilisé",y= "log_median DGCI")+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())
bxp
bxp <- bxp + facet_wrap(~ asso_stade_IA, ncol=3)
stat.test <- stat.test %>% add_xy_position(x = "asso_stade_IA")
bxp + 
  stat_pvalue_manual(stat.test, label = "p.adj.signif") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.10)))
#
#
#

