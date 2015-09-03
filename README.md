# DataPopularity
## Introduction
The LHCb collaboration is one of the four major experiments at the Large Hadron Collider at CERN. The detector, as well as the Monte Carlo simulations of physics events, create PBs of data every year. This data is kept on disk and tape storage systems. Disks are used for storing data used by physicists for analysis. They are much faster than tapes, but are way more expensive and hence disk space is limited. Therefore it is highly important to identify which datasets should be kept on disk and which ones should only be kept as archives on tape. The system presented here is designed to select the datasets which may be used in the future
and thus should remain on disk. Input information to the system are the dataset usage history and dataset metadata (size, type, configuration etc.).

## Method Descriprion
The method was presented on [CHEP2015](https://indico.cern.ch/event/304944/session/3/contribution/303/attachments/578882/797086/DataPopularityPresentation.pdf). This presentation contains the method idea and its parameters description. Please, view this presentation firstly.

## Structure
**Datapop** is the python library. This is a realisation of the method of the disk storage mangement for the LHCb.

**Datapopserv** is the data popularity servise for the method. This service can be launched and can be used instead of the datapop library. Datapopserv uses curl requets.

**Datapopclient** is python wrapper for the data popularity service. The wrapper provides the comfortable way for working with the service.

Look at [howto](https://github.com/yandexdataschool/DataPopularity/tree/master/howto) for the more details.
