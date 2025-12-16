library(tidyverse)

## Error indices

setwd("~/Documentos/repositories/datos_Alberto")

list.files("./out/errors/partial_fit_1/") %>%
  set_names(., .) %>%
  map(., ~ paste0("./out/errors/partial_fit_1/", .)) %>%
  map(., ~ read_csv(.)) %>%
  imap_dfr(., ~ mutate(.x, cond = .y)) %>%
  separate(cond, into = c("model", "order"), sep = "_") %>%
  summarize(AUC = mean(scores), s = sd(scores), 
            .by = c("type", "model", "order")) %>%
  mutate(s = ifelse(type == "test", 0, s)) %>%
  ggplot(aes(x = model, y = AUC, fill = type, colour = type)) +
  geom_col(position = "dodge") +
  geom_errorbar(aes(ymin = AUC-s, ymax = AUC+s), position = position_dodge()) +
  facet_wrap("order") + ggtitle("Errors after hyperparameter tuning")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  

## 

aa <- list.files("./out/predictions/partial_fit/") %>%
  set_names(., .) %>%
  map(., ~ paste0("./out/predictions/partial_fit/", .)) %>%
  map(., ~ read_csv(.)) %>%  #mutate(., Aw = 1 - Bw^2)) %>%
  map(.,
      ~ filter(., abs(.94-Aw) < .01) 
      ) %>%
  imap(.,
      ~ ggplot(.x, aes(x = pH, y = Temperatura, fill = factor(pred))) +
        geom_tile() +
        ggtitle(.y) +
        theme(legend.position = "none")
      ) 
  
title_1 <- cowplot::ggdraw() + 
  cowplot::draw_label(
    "Decision boundaries hyperparameters tuned I",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )

title_2 <- cowplot::ggdraw() + 
  cowplot::draw_label(
    "Decision boundaries hyperparameters tuned II",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    # add margin on the left of the drawing canvas,
    # so title is aligned with left edge of first plot
    plot.margin = margin(0, 0, 0, 7)
  )

plot_1 <- cowplot::plot_grid(plotlist = aa[1:12])
cowplot::plot_grid(title_1, plot_1, ncol = 1, rel_heights = c(0.1,1))

plot_2 <- cowplot::plot_grid(plotlist = aa[13:20])
cowplot::plot_grid(title_2, plot_2, ncol = 1, rel_heights = c(0.1,1))


## no ejecutado

bind_rows(read_tsv("./data/d1.csv"),
          read_tsv("./data/d2.csv")) %>%
  mutate(., Aw = 1 - Bw^2) %>%
  summarize(p = mean(Crec), 
            .by = c("Temperatura", "pH", "Aw")
            ) %>%
  ggplot() +
  geom_point(aes(x = pH, y = Temperatura, size = p)) +
  facet_wrap("Aw")
          





