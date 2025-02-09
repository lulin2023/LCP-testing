############ Two sample conditional distribution test -------------


ModelA <- read.csv("ModelA_500times.csv")[,-1]
ModelB <- read.csv("ModelB_500times.csv")[,-1]
ModelC <- read.csv("ModelC_500times.csv")[,-1]

# ModelA <- read.csv("ModelA_20.csv")[,-1]
# ModelB <- read.csv("ModelB_20.csv")[,-1]
# ModelC <- read.csv("ModelC.csv")[,-1]

Result_all <- ModelA 
Result_all$n <- factor(Result_all$n, levels = c('200','400','600','800','1000'))
Result_all$type <- factor(Result_all$type, levels = c('Type I error','Power'))

Result_all$Method[which(Result_all$Method=='weight')] = 'LCT'
Result_all$Method[which(Result_all$Method=='ori')] = 'CT'
Result_all$Method[which(Result_all$Method=='debias')] = 'DCT'

Result_all$Method <- factor(Result_all$Method, levels = c('LCT','CT','DCT'))
pp <- Result_all%>%
  group_by(Method, n,Alg,type,hline)%>%
  dplyr::summarize(Quant = mean(quant, na.rm = TRUE), sdQuant = sd(quant, na.rm = TRUE))
pp

class(pp)



p1 <- ggplot(data = pp,aes(x=n,y=Quant,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.0)+geom_ribbon(aes(ymin = Quant - sdQuant,ymax = Quant + sdQuant),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  scale_y_continuous(limits = c(0, 1),
                     breaks = seq(0,1,0.25))+
  xlab("n")+
  ylab("Scenario A3")+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF"))+
  facet_grid(type~Alg, scales = "free")+
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 

p1


Result_all <- ModelB 
Result_all$n <- factor(Result_all$n, levels = c('200','400','600','800','1000'))
Result_all$type <- factor(Result_all$type, levels = c('Type I error','Power'))
Result_all$Method[which(Result_all$Method=='weight')] = 'LCT'
Result_all$Method[which(Result_all$Method=='ori')] = 'CT'
Result_all$Method[which(Result_all$Method=='debias')] = 'DCT'

Result_all$Method <- factor(Result_all$Method, levels = c('LCT','CT','DCT'))
pp <- Result_all%>%
  group_by(Method, n,Alg,type,hline)%>%
  dplyr::summarize(Quant = mean(quant,na.rm=T), sdQuant = sd(quant,na.rm=T))
pp

class(pp)



p2<- ggplot(data = pp,aes(x=n,y=Quant,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.0)+geom_ribbon(aes(ymin = Quant - sdQuant,ymax = Quant + sdQuant),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  scale_y_continuous(limits = c(0, 1),
                     breaks = seq(0,1,0.25))+
  xlab("n")+
  ylab("Scenario B3")+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF"))+
  facet_grid(type~Alg, scales = "free")+
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 

p2



Result_all <- ModelC
Result_all$n <- factor(Result_all$n, levels = c('200','400','600','800','1000'))
Result_all$type <- factor(Result_all$type, levels = c('Type I error','Power'))
Result_all$Method[which(Result_all$Method=='weight')] = 'LCT'
Result_all$Method[which(Result_all$Method=='ori')] = 'CT'
Result_all$Method[which(Result_all$Method=='debias')] = 'DCT'

Result_all$Method <- factor(Result_all$Method, levels = c('LCT','CT','DCT'))
pp <- Result_all%>%
  group_by(Method, n,Alg,type,hline)%>%
  dplyr::summarize(Quant = mean(quant,na.rm=T), sdQuant = sd(quant,na.rm=T))
pp

class(pp)





p3 <- ggplot(data = pp,aes(x=n,y=Quant,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.0)+geom_ribbon(aes(ymin = Quant - sdQuant,ymax = Quant + sdQuant),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  scale_y_continuous(limits = c(0, 1),
                     breaks = seq(0,1,0.25))+
  xlab("n")+
  ylab("Scenario C3")+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF"))+
  facet_grid(type~Alg, scales = "free")+
  geom_hline(aes(yintercept = hline), colour = "#AA0000", na.rm = T,linetype="dashed") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif")) +
  theme(legend.position = "bottom") 

p3
dev.off()

pdf(file="Two-sample-test.pdf",
    width=10,height=16)

P <- ggarrange(p1, p2, p3, ncol=1, nrow=3, common.legend = TRUE, legend="bottom",
               font.label = list(size = 20, face = "bold"))
P
dev.off()
dev.new()
