Instruções
Na aula passada, discutimos que usar IA apenas para gerar código não garante que um sistema esteja bem arquitetado, seguro, escalável ou tecnicamente sustentável. A IA pode acelerar o desenvolvimento, mas o papel do desenvolvedor continua sendo essencial para revisar decisões, validar segurança, corrigir acoplamentos ruins e transformar código gerado em um produto real.

Com base nisso, cada ser, aluno ou louco,  deverá revisar o projeto desenvolvido até agora, analisando se a arquitetura atual realmente suporta crescimento, manutenção e integração com novas funcionalidades. O objetivo não é apenas “fazer funcionar”, mas entender se o projeto está bem estruturado.

Parte 1 — Revisão da Arquitetura
deverá verificar se o projeto possui uma divisão clara entre:

frontend;

backend/API;

banco de dados;

serviços internos;

camada de IA/modelo;

camada de integração externa;

camada de web scraping.

A análise deve responder:

A interface está apenas exibindo dados ou também possui regra de negócio indevida?

O backend concentra a lógica principal do sistema?

O acesso ao banco está isolado em uma camada própria ou aparece espalhado pelo código?

A chamada ao modelo de IA/YOLO está separada da regra de negócio?

A nova camada de scraping será implementada como serviço separado ou ficará misturada em rotas, telas ou controllers?

Parte 2 — Revisão de Segurança
deverá identificar riscos de segurança no projeto. Alguns pontos para observar:

Existem senhas, tokens ou chaves diretamente no código?

As rotas da API estão abertas sem validação?

Os dados enviados pelo usuário são validados antes de serem processados?

Existe risco de SQL Injection, exposição de dados ou acesso indevido?

O sistema trata erros de forma segura ou mostra mensagens técnicas demais ao usuário?

Caso o projeto use IA, scraping ou upload de arquivos, o grupo também deve avaliar se existe risco de processar dados maliciosos ou fontes não confiáveis.

Parte 3 — Melhoria do Código Gerado com IA
Todo o código.

Para cada trecho, o  aluno deve apresentar:

O que o código fazia originalmente;

Qual era o problema encontrado;

O que foi melhorado;

Por que a nova versão é melhor.

Parte 4 — Implementação de uma Camada de Web Scraping
Agora o grupo deverá implementar uma camada de web scraping no projeto.

A ideia é buscar dados públicos e gratuitos na internet para enriquecer o sistema. Essa camada não deve ser feita de qualquer forma. Ela precisa respeitar boas práticas, evitar excesso de requisições e tratar erros corretamente.

No caso do projeto AgroVision, por exemplo, se o sistema identifica pessoas, veículos, animais, máquinas ou movimentações por YOLO, a camada de web scraping poderia buscar informações públicas complementares, como:

previsão do tempo;

cotação agrícola;

notícias do setor agro;

preço de commodities;

alertas climáticos;

dados públicos de mercado;

informações de safras ou logística.

O grupo deve justificar qual dado será coletado e como ele melhora o projeto.

Requisitos Técnicos da Camada de Web Scraping
A implementação deve conter:

Uma função ou serviço separado apenas para scraping;

Uso de fonte pública e gratuita;

Tratamento de erro caso o site esteja fora do ar;

Limite de requisições para não sobrecarregar a fonte;

Organização dos dados coletados em formato estruturado, como JSON;

Integração com alguma parte do sistema, seja banco, API ou tela;

Explicação clara de por que aquela informação é relevante para o projeto.

O scraping não pode ser apenas “copiar dados da internet”. Ele precisa ter uma finalidade dentro do sistema.
