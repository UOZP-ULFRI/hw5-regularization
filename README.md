# Kratka domača naloga 5: Regularizacija (UOZP)

V tokratni domači nalogi bomo implementirali linearno regresijo z analitično rešitvijo (angl. *closed-form*).
Primerjali bomo napovedi modela z implementacijo gradientnega sestopa iz prejšnje naloge. Kodo iz prejšnje naloge lahko prekopirate in jo nadgradite za uporabo regularizacije.

---

## Python okolje

Potrebujete enako python okolje kot pri prejšni nalogi, torej Python 3.8 ali višjo verzijo.
Za implementacijo je potrebna le knjižnica **numpy**.
V kolikor si še niste, si ustvarite virtualno okolje za delo, kot zapoveduje dobra praksa.

---

Skozi implementacijo naloge vas bodo vodili testi, zato preberite opozorila, ki se izpišejo v konzoli. Kode za to nalogo je relativno malo, vendar je razhroščevanje lahko dolgotrajno. Predlagamo, da dele svoje kode še sami testirate.

## 1. Implementacija linearne regresije z analitično rešitvijo

Vaša naloga je implementacija zaprte oblike linearne regresije.
Enačbo ste izpeljali na predavanjih, končno obliko pa vam podajamo tu:

$y = X \cdot w$

$w = (X^TX)^{-1}Xy$

V prvem koraku implementirajte rešitev v vektorski obliki. Dodajte jo v metodo `fit` razreda `LinearRegression`. Metoda `predict` naj vrača napovedi na validacijski/testni množici.

## 2. Implementacija zaprte oblike z regularizacijskim parameterom

Rešitev iz prejšnjega koraka nadgradite z dodajanjem L2 regularizacijskega parametra v enačbo.

$w = (X^TX + \lambda I)^{-1}Xy$

Pri tem pazite, da ostane konstantni člen (angl. *intercept*) brez regularizacije. Torej vrednost uteži ne bo prispevala k višji vrednosti cenilne funkcije. 

*Namig: Potrebno po prilagoditi matriko I.*

## 3. Implementacija gradientne rešitve z regularizacijskim parametrom

V nalogi 4 ste implementirali rešitev linearne regresije z uporabo gradientnega sestopa. Večino kode lahko prepišete in jo shranite v podane funkcije. Linearna regresija s sestopom je podana v razredu `LinearRegressionGD`, ki ima ponovno funkciji `fit` in `predict`.

Za lažjo orientacijo vam podajamo cenilno funkcijo. Obstaja več variacij funkcije in njenega odvoda. Mi smo izbrali tako, da bosta parametra L2 regularizacije ($\lambda$) ekvivalentna med implementacijama. Tudi testi pričakujejo, da sta parametra $\lambda$ ekvivalentna.

$J(\theta) = \frac{1}{2} \sum_{i=1}^m (f(x^{(i)}) - y^{(i)})^2 + \frac{1}{2} \lambda \sum_{j=1}^n \theta_j^2$

Gradient izpeljite sami. Ne pozabite, da morate odvajati tudi regularizacijski del.

Pri enem izmed testov je parameter $\lambda$ nastavljen na visoko vrednost, ki bo problematična za optimizacijo s sestopom. Razmislite kaj je problem pri teh vrednostih in poskusite spreminjati stopnjo učenja v takih primerih dinamično.

## 4. Izbira najboljšega L2 parametra ($\lambda$)

L2 regularizacijski parameter je hiper parameter, ki ga je potrebno optimizirati kot ostale parametre modela. Najosnovnejša metoda za oceno najboljšega parametra je izračun napake (npr. MSE) na validacijski množici, ki je nismo uporabili za učenje.

Implementirajte metodo `find_best_lambda`, ki sprejme učno in validacijsko množico ter seznam kandidatov za parameter $\lambda$. Metoda naj uporabi linearno regresijo z analitično rešitvijo.
Za vsak parameter morate ustvariti nov model s tem parametrom, ga učiti na učni množici ter izračunati napako na validacijski množici.

Metoda naj vrne najboljšo izbiro parametra $\lambda$ glede na napako na validacijki množici. 

Testi za ta del naloge pregledajo različne tipe podatkovnih množic s 50 značilkami. Za vsak test se vam bo izpisal tekst s parametri množice, vi pa premislite ali je izbira parametra smiselna.
Posebej bodite pozorni na velikost učne množice v razmeru z značilkami (2:1 in 1:1) ter količino šuma v podatkih (nič, malenkost, nekaj in ogromno).

