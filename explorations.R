
library(tidyverse)

## Error indices

list.files("./out/errors") %>%
  set_names(., .) %>%
  map(., ~ paste0("./out/errors/", .)) %>%
  map(., ~ read_csv(.)) %>%
  imap_dfr(., ~ mutate(.x, cond = .y)) %>%
  separate(cond, into = c("model", "order"), sep = "_") %>%
  summarize(AUC = mean(scores), s = sd(scores), 
            .by = c("type", "model", "order")) %>%
  mutate(s = ifelse(type == "test", 0, s)) %>%
  ggplot(aes(x = model, y = AUC, fill = type, colour = type)) +
  geom_col(position = "dodge") +
  geom_errorbar(aes(ymin = AUC-s, ymax = AUC+s), position = position_dodge()) +
  facet_wrap("order") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  

## 

aa <- list.files("./out/predictions") %>%
  set_names(., .) %>%
  map(., ~ paste0("./out/predictions/", .)) %>%
  map(., 
      ~ read_csv(.) %>%  mutate(., Aw = 1 - Bw^2)
  ) %>%
  map(.,
      ~ filter(., abs(.94-Aw) < .01) 
      ) %>%
  imap(.,
      ~ ggplot(.x, aes(x = pH, y = Temperatura, fill = factor(pred))) +
        geom_tile() +
        ggtitle(.y) +
        theme(legend.position = "none")
      ) 
  
cowplot::plot_grid(plotlist = aa[1:12]) 
cowplot::plot_grid(plotlist = aa[13:20])

##

bind_rows(read_tsv("./data/d1.csv"),
          read_tsv("./data/d2.csv")) %>%
  mutate(., Aw = 1 - Bw^2) %>%
  summarize(p = mean(Crec), 
            .by = c("Temperatura", "pH", "Aw")
            ) %>%
  ggplot() +
  geom_point(aes(x = pH, y = Temperatura, size = p)) +
  facet_wrap("Aw")
          





