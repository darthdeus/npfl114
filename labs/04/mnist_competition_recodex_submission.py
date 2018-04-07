# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        gpu_options = tf.GPUOptions(allow_growth=True)

        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads,
                                                                     gpu_options=gpu_options))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder_with_default(False, [], name=\"is_training\")

            # Computation
            # TODO: Add layers described in the args.cnn. Layers are separated by a comma and
            # in addition to the ones allowed in mnist_conv.py you should also support
            # - CB-filters-kernel_size-stride-padding: Add a convolutional layer with BatchNorm
            #   and ReLU activation and specified number of filters, kernel size, stride and padding.
            #   Example: CB-10-3-1-same
            # To correctly implement BatchNorm:
            # - The convolutional layer should not use any activation and no biases.
            # - The output of the convolutional layer is passed to batch_normalization layer, which
            #   should specify `training=True` during training and `training=False` during inference.
            # - The output of the batch_normalization layer is passed through tf.nn.relu.
            # - You need to update the moving averages of mean and variance in the batch normalization
            #   layer during each training batch. Such update operations can be obtained using
            #   `tf.get_collection(tf.GraphKeys.UPDATE_OPS)` and utilized either directly in `session.run`,
            #   or (preferably) attached to `self.train` using `tf.control_dependencies`.
            # Store result in `features`.

            features = self.images

            for layer in args.cnn.split(\",\"):
                layer_args = layer.split(\"-\")

                type = layer_args[0]

                if type == \"CB\":
                    features = tf.layers.conv2d(features,
                                                filters=int(layer_args[1]),
                                                kernel_size=int(layer_args[2]),
                                                strides=int(layer_args[3]),
                                                padding=layer_args[4],
                                                activation=None)

                    features = tf.layers.batch_normalization(features,
                                                             training=self.is_training)

                    features = tf.nn.relu(features)
                elif type == \"C\":
                    features = tf.layers.conv2d(features,
                                                filters=int(layer_args[1]),
                                                kernel_size=int(layer_args[2]),
                                                strides=int(layer_args[3]),
                                                padding=layer_args[4],
                                                activation=tf.nn.relu)
                elif type == \"M\":
                    features = tf.layers.max_pooling2d(features,
                                                       pool_size=int(layer_args[1]),
                                                       strides=int(layer_args[2]))
                elif type == \"F\":
                    features = tf.layers.flatten(features)
                elif type == \"R\":
                    features = tf.layers.dense(features,
                                               units=int(layer_args[1]),
                                               activation=tf.nn.relu)

                    features = tf.layers.dropout(features,
                                                 0.5,
                                                 training=self.is_training)

            output_layer = tf.layers.dense(features, self.LABELS, activation=None, name=\"output_layer\")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")
            global_step = tf.train.create_global_step()

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(loss,
                                                                                      global_step=global_step,
                                                                                      name=\"training\")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {
            self.images: images,
            self.labels: labels,
            self.is_training: True
        })

    def evaluate(self, dataset, images, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {
            self.images: images,
            self.labels: labels,
        })
        return accuracy

    def predict(self, images):
        return self.session.run(self.predictions, {self.images: images})


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=128, type=int, help=\"Batch size.\")
    parser.add_argument(\"--cnn\", default=\"CB-128-3-1-same,M-3-2,CB-128-3-1-same,CB-64-3-1-same,M-2-1,F,R-1024\", type=str,
                        help=\"Description of the CNN architecture.\")
    parser.add_argument(\"--epochs\", default=20, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=12, type=int, help=\"Maximum number of threads to use.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\")  # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist

    mnist = mnist.input_data.read_data_sets(\"mnist-gan\",
                                            reshape=False,
                                            seed=42,
                                            source_url=\"https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/\")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)

    label_groups = []

    bs = 2048

    test_imgs = mnist.test.images
    N = test_imgs.shape[0]

    for idx in np.split(np.arange(N), N // bs):
        test_batch = test_imgs[idx]
        label_groups.append(network.predict(test_batch))

    test_labels = np.concatenate(label_groups)

    np.savetxt(\"mnist_competition_test.txt\", test_labels, fmt=\"%d\", delimiter=\"\\n\")
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;B_w+VO;<j13gV=7Yq_8Jt(?mHZYog4pqV6G^cyK9Szyu=and!)1{nG33dLhx%)xq>_3>kdU~a)(0^-?KQi8Nnlbf)8EbDt5U9STBxaRafP_ak9`i;tKO!3gJ&#<m<pHT5!H=)dDcm!M9ozOA{XDR`)3p!{b6aMGG4jy`z88h@64AlWrWeeH3JNibP>xJ^Jl=i(jQ)EN4CT~>7}WQRH~bWkh&SqkTk^WBcoe^7B=iK7MTPmVr(_M}-5V)ETNA~`cI>sg?qYsP<|e_~61Q&6JMxm84s{N`wq<6A5{xyu1~Oo)ZJg^sTM)X!B4H?PGW6B>G0dtORXNwET8F1BENThdkgt2r$st7j^fwpV@F3NU^393$F9DuoF52+lpPYYI(%Xq;^J{?I^GV2b%Yw7%lxsf)_the2bIsb#bveW`9=_^HCvB%9mnI8|K9rMI$TATxDkHHOUUKCy9%qhSByrV?-}M&95vhjc4rCe{I?9JcHnLF1yi61>=tvYuWg7UW9I1wn?wv4-lZi9d?9VsmXdE3<xC=NGo@pe!8JU}ww{BL@Ut6nh=*PxB^LRG#hTq?lg+=7d<Q`N{_8kq)RA(01cx#fj4*zaicYLz}?5!e68&=@NEc4fMtM@@vK0H?(%LsyNOKp&4EUUCd&_NBHB3>hNL7X-ho^g_D+qp)Vr_e{54c<HVyYn@XdbW7>%_lzc!A8P$$dq>Y&nZh7A)=y;gcn9%R0?5NF~uq31>1~JtCLb!uVJ+s3S$7Fh9tNr4i)32K%2FB&h;=kM7bXpl9zsvFXQNf)-kx0k3<Z>bL`6^PL`DfN6=yXj)6F&)mwFet|FvogfZPqJ?u4ozO@`geu_=4ek?%*k&CDUg8vZqC}c;##j_8Mt+pQ`%YkJ=94T=^)#iq$I#1ntfh9yOx{7!^-p%CQ{UoJHCB@o&%7M{`ifgy@1P#~MiQVq_l_$9n&O|*2MHm_S7-Xvi*yh|ah{oSdx8wl1Z(eM_SN(a3c5(JY8SCr@u1~57Q#{4cD5ZR4%^?(BpLA!!`&!qq!rR|TKrO=n4p&#D*P_Gf<cEEwav=#vFrQk`r@-OCRjz0GH5rt5-jviB5A(gXvzm}$rt4+O0oxK>utF5P$m0gxV3!AMgmhVX69t(M0!)T9Y!S(17SYi*x*rmU(jhC4)<HqUcor<2&sMfKUXuaJQzehI&cB>X{lpY9yAdTRd${eEb64VQ*N4E>V~XBc*MAq^Cz;x>yuj0yB^}+VT3oLiG*Bk#PZ7bSp;r!>NsQQ`fn2gvcS9F1X$Xh;nFg37wIGInG>=%hWIU1lrW?Cy6ddq!8ZpaH4m~ey*C{OmlB$BGBB<YC<6qI5h%@y_Rwr*><(*ek03SDqDY#ZBK`IEFF|)3m2buOa29|ye=ZZTnP&-#_U#=5c{gkrR9lWjl$=JA22(|t}4Mrs0XNWmZvV;XU7@*47vlt50m8FR4WDG3opE$JlM|s_$GzjzOfzPlQVJitWdLDct@RY->zJDk|VPb^WHou8QBKkQ%+FhjGyNkLgJU46OHwV`tPP==_2(Y$2&mr~f99q5_Q0pXiAFAtqodvcMQphZEkBLO<Dsos3m!fk(bX*be+Or&3ZHs%++}bkx{BL78vVV(Iw&T$0;l(S>gZ1}6<?pF@)9M`zbyKavWT84NPo9U*MGX&{Bk*U?BCi7!Ot!f=%!PNh`Rb^fH#m{LSK@Z1AK%1}%K*V3D|qe2$@KW8^j7{EsM!qC_P0CmUdT~rM@@H#ARvrUtRi*GQ(I565MJ;fZn!9b>1#f`saHB&<Q*Xpi(Yb|K`Y(zmqh4+of8O0(_Up?j#&RciI!;67%ukieLo{bJL=%%HD_BMyMq$3EaS>HAx8d74nl1T)iBMHA4N*p0>vpH0}V{|Jy>KtGPPrIX7M!_mQu?}Hgm5{)M`ZA$((fEwnOdWJ~MJ7VfD;D*0S>OO&W9>fxVDkm1UlzDHg=2Dz^iItwcWpH3*Pphy(y~>_JaN*fMi<rq8ET)63(4lz%Ij=`^4A7yk)3+2)t%dmx;Z*iA<)&TkW{qn0zg?D`f}aTIs_ENJwK^d>E#D7>m~&DWRr-9Vc1rSqa{`!iW*fI#QQC6%;twibTyELuymFpv63w1AIbykD5`0<B!RXdg1-@pP8zD<SZ?&5NVNnxvHUHbf5ePnQT%>)umPt40f@T^P@<l)|UyE_aP5p_`Mr%+TQxCe)eekkDq*9o!?yVOBF7vBEUG#d1FSK%FtK1AfLi$&C&Vj|)GZ9HYu*6INVrC5wA@g<+sj*+Eg^1<zqM1oCYB|8YW3sRA6Xop<#BKJuzfrUnnA^b`IM=>m)YUuxMN0{S8`(o26}R<r_Idj9bMSRHYrEzM9V{Bp6VC%Aj<>o*XVfeXn+g~mUZqXaz6BH79II{nRbxJNa57u<Z9T>bbVtNOyM4zYgM-I$ImiVWFY!co<V-t6cnL&LrIwoHC%%Age7@+LUM+S4hoetxo>Fz5M4Nsz^b1@mh!7oW!^Dfl%!Xdv6u(bW#EJp7QY@ZV`n7qdHvV3h^1Dk{+LBFjoD^C8soP2p+*B7DI}cXVid6{xvH;T}5Ip%hwMdowPyhs9$~fEbVvTW)@4tFd%2srOnb-CGu!4x)44QX{RnX8`%7QD&18`bxnCD9;Bxb|3cT@(|*mOrnBF{xy<s+v>*8ngJ`E$}zB2P+@kNqs9wy=t7N%y>oLmzio>6nob({kv@Qj(U7+TIP~d}*9hH2!G9agi0Nn^-FtpvGMW&#pJ349&zH>{#jrSnk&{qF`mk-x9kJ~<@zyH>#-bn0a>Q@VIKVwW-jtNY5u|i#Y@aF|w_8|`UXIqs)o>W@IML}D6t*wrX`I((`A?KpW*o##5|jz|UNrwXbL}7{)go*TB0ewdW1JPLCv$(LhIoyLGQ}tSO=p{m#@4rnbfnkUr~>1IrE#)}4AQ>O!X{51Q$EzT>ysA8a`xj?`KL{Q{Z=j0mE-F)WFa#<j+_|39waGuYsp)VTZ%#(no`jC8$Y~(<{Yvl&}N!2M5qZe5tKa{QNRd|dMsdkaAyi9j~B^2YeS;6I`s_aIw`5$77k=}TAu$NKZdrzf6jOQTS4tJzVrY5oAib$9cQhmA7Cg~AKzs67ji^iCJ7mg#9{jS5+emZsPkFV1iKYI2eyAEFAE+%HV;?}iS@34Pa;HIA|EAz{&N<0NCmrH=fvKj)jQoR5=Su5&wYw1o(<G6BDZ~&b3f=nPaEQ7s4F20Y?dm~U*(5eE{f^+U;FgZ8Vuf$yr|dv#-^OmK}cUFOq!rmQ*7k>y5D;2jJ4FJKmjuy$T@FKV2XMm21Kkan^^#4_A1Tlq@1C)lEUFR3T?`Dm2C#vo7~vHMA7a49S0+-uH6J}PKEJ;p-glYnuvcdeb0JHT1c#fIgmkFtnXoS2-wai5SlCDfFwZrR&S2AXjSm*&ST{A)gy9+=uCDSPLf2DljfdQAnnjbv$Co5;3?Sf<=$5iVF@yf5%&d!tj+3}!QU(Mc|b?Q@GS@dW_7AjLk3Q-IJrn{OWk72$DS};ps>Ara3xvel^GLDrt(q|(m{jN^+)5Sh+{I!zpgh6&ob&k-vIlXk!%_*EjJd#O!_i2<D5UhB^imrjzNY7B#6x7`!Us)c|~&uNabyypn8D#jfDdM5x%3w;7zTNrm<r~e@a25qcwV81Po*-sTxMnc8xy`^u($?r&C?B15-&JcF0uuG{|#3mgQgjl;0!UW#7vh80K?-=e5QGnlW@%3*v~@+JiManZ>XtD+jbhm|4}@JAvT6K<tXD%=EI3M~OY{l0}i6n@!9=3*f`L9%Zcy&)sX4brisCwawK6_$YtZFq)oJnxM)f1nS?gZ3@;WZ-_@h(7qQHzZY{Mfq~(k6tO&39QHf)hN<|9weXM*o%t{II-xC)tYaL6!B~^7D0G=tUHD)7syJEFLEps~%J2GbgS>REgu2oVW4OZVUdyUekY8iyypeGzy^O5>Fr{3N%gU~wdAE|j8I^3&TmcJwjE1%InKt<%O_#N&@k1LTbY#s2z5bV9L?GK>ffBnSNu9@PXVWDNXs=}Z3{5TFsLrAYY`04i3Uu>#_XYhuds#X(qf#4;pk_7aLQXJi8}OIPORleQte-DoqC^ZOz6dm2|03aTE(axl`EvdgVR>_xylr5=dGwlI=dilUmxvO5vaF^3ii3oYO5@kVsdGtfhOlD`(kd<E5!ewz3SNKq=mta>SZkfb)rym`nr1iKVqUGF^MtR8>u2~(z!wL2&fqn9R<GMp_Tn#Qw#JXy-$MQ7bj|%K1r*t)cteboG$0*vUO60d$u%${OL<W#83(&xKOFhf<ViSi8uZDHH<rkK+tUzEc{1|H*=|OF0Jbb+E_q64s*@NLlC+YiP8bc@de^x5G>5}ZeNucg0P|06;DZfQaBMU|s}zH@$%{Wfj_vf5h0cE+0R|B^f$K#AZDH0zkb;j+B1!7>4pdOVB*D<2^O(tB@FDU?MFD4K#p;*uj#mqldXv+Tq%|JgmSTZP97;Il+{<$jeq6!ajGxS(Y4QsEpa`b$R=hgdn<Z9*`gw7|2lj@h1c7If%X{0ZTm_T>mN`c6diT!CJXgjeR$sj3zm&qQJ!(^Yz6moAx!^7xRq&egil6M=-4IoS?0v1XL$qqb8XEp>y_`igYrb9a0cQ>E02+fc(rZL;zd!;-C0ix?sI(L8c&czTv5flGhDZlKXg<k_B`Nh#@Xg*57INnXBZ}`36Rbl0tw<3EUL(Skd!pWayy3nsOZ@~xkn~(bZ)7U@Z-8utg&8qKIq}_mQnQ^R+XbIH-ZK}Lp>{nU0W47&aTFYPI<zl@#xFyS)}p3y*|aT$H+YL1qTABUN62PKRz_x%(it-u*-9m$+HB_|QJkPCO0)w{=5JQqmxe~H>H3R`ExJ6f((O8khy9d3tX4CVBZ-Z;9|eq9`j*z^k$v8?7tf6RneQ5n-ty8Q4qwr!micRn;Nk5l>Ocd^*0TMmB~!R~{03ggMCK#HIT-5nWdNR~GFEgG(hz;NEe2WBZ&)U)Qw~b$3SqvMARWWJjBu05b~8N*fB}PDw>NOBr78iyseh+v=%in%RIGn}lyF?xiWpx@J$1{uGj|3ADP1N)sw-l~oI^T8Z$;}#*bCok=jqG;V`=`MQRrq$o8}i_9Vmil_H;=#NE+)UL2@`293$jLn_&hTKb`gk;?z}mzVTkFk!p{j<SEIXr6T4bs-Cc39F=sc7C<Bs@ax3wlb;|~m^u|FLQKu&b?azJ1k!q-8reDlGYW!x3aTv0-*%^2K*Yk;W_zW`+0=Wg-*gIHS&&9DPM%!&`fx^(?752$gw*ESTxSp8FHyi~ZxD;Y<F{RGAs@UGkNY!_+Du#E;ec4;&rwTbyLOOLW3-}EU`LDQi=IV%42)0CCb~bEFv0h=!)7xEg$qeQD>LFUwVxcws(ZIowpSSaqOZGgE7{LyS8Yv4gLPazi?M3>m@RK~0F-o`TaXyG(PJu8tG1RgYc(Mngjt<SI=WIBbv#Y{tnhX@z?drb`)5&n$z8FQi=;?eH5!zi8mjIeKEy%S*31&>%&}HLF-?OIcZGf!*Aj|x0mj6i;=3^7hEE!pc!DdCePJ%7q9I2?2*g~_E#}7vt{BZnMSpam_qkTHh{y+g9xEGX=u(eh{$2ny)f#I?Z%fwQ6<TcKKL3k=33JvEsx~!uoAQg+caqd+a_R~$zWkj*26*>V7x{&3k^6*Ne?6VhTt}-Viqzc$5&)Ir7f89N3fSM$45u`A4Jo>Nul$n3x)gT+0p9(iQUK8my-Jvn_JJ;H1IEyz&W@Jr`-nqeEp&yK&PMY@W=pcAsMv($?08Xw>1Y14RE0%b+OP$$x0FEXI94*xmPTkw6JvSZvmgY6wi4p7-+8maVG%Wg$0mgfDZi?n#S4Rtt>sT!lZCsXJ&^7UqWz8)u?buqBic2?ikw%%PW*|@VuvoMWrDh>U4za$rZ>5e8#R&VVSHx?niCr~C@VhxCDFnj<md*$EB$Zmi-dpOMRI+@A@EEq2T~qYz#G)qG|Ao2kS_e(029T&_51570b|(_e7T7}I`rsf{u}NX&HX`?V(GT<ji|%@klQpid`pLjYYpH~ga(K4$Kaj}XQ>BRi4Fpgc>*cV!7Jst3Lm1uKdff|(F<*=VCt@rj>at-Hg(-I22JCn-E~gzhAx%2D}~P^Thi$DU-Nc7;)^KV&CD4v9Xzdv$dX-}p4oli&(-67L}6g})|SV(r+bgu&yT4kn^}`Gt%UViv-&HAa*Q2ho5La1LJgzRSJd8I=84++-cE#xx`uc947jt)@)Wljc?ECeMP9)EJvYzKv%VbhF3o*4mb0cDvkT79Y>X3q?D_lH00q>oW#M133=?W<6a2QTgu+Js9q-MFAOxN(H3~wdXnOAs*y1OkR~&rDrH<i6;xMG{M=m_5If+IZrh4{OaZhC)U9xS%AI`(e>sK(!+vDxCZ27`uAO{>np3y7dPlH!-cO!USLBo&TL<o_L-Z&<n6&f#>e%d#o#_j}@QvPM|RK6;X_W2Dz-Utvva&(sXEm4~pWL0INsMkDE&q+NMz;~idL5}@hTxl|{#^T!tllv-h7N`oU7Uu!OD|F<JqA}mXtfWW8lCcAOy~ibTNqngr0WPH0ldopv_31<$JgZkoG321dSiAD8EG|L4_EHksrnhlLMwd&GpqWuhWVJuCU3acX_JA|l-=FQ(?vA^L!hNn<J5p1i@}xjeFGt~8(_~DV_`-|DrFc<Xygp!2?_oiNWotX=9pr1+!M6P`$=Z&_D}}G1Kgt^VL7ZfP@9n3tNLL@Y)VJeyeq5g4<6|ALv_pzOEz(MkA)}nM#<0D{@S>=b!W6Gz;>D=_UKB|!0Kfvn&vRzZFGVC_rNSZ|=kDvj@69!LZE@@12&0=zIdvug7X@<IZ*5z87Iop`Q+dKmB7l^hfM9FJHngdiZ;Y>0F&LYK8Sl%0A}F-;3|>{s!oM)Crt(hnW#3gGf2s`ACr&%{#hy4($idKGS$zvEDHzl>Cy&e>MRZ`^6FvQ@*|mpHr@!BRw5c$jY#Mr-Cn`Dcn3RlT(+;o8<0><sujNJ4FURet9!SUg-XIy&PhTnU(BRecXk*VfLjT75XomhjOs{;t?j^1%OvCsI_OcQWjLNydfDxu(zBi_*uD>2#>I{LsIn}@@C;~7ct4LuhXtZ)HNZ!tH-P;M=qfAx)AS=1+yVX7IH0>vJ@=0{lsdLepQ@9X`NvHB?pJ6FwvpR2jFKTRS?$a(-kGt@P$F9XS-SrqgsQMmqg^|I?cAcHUIV@UyTTTLC>{ywkdz>b_aRGb6{y+zo2C15D(jv?B00000^Ib}66T*m600I3hu<8K-vNm+ivBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
