function ProjectDisplay(props) {
    return (
        <div className={props.id === 0 ? 'carousel-item active' : 'carousel-item'}>
            <div className="img-container">
                <img className="d-block w-100" src={props.link} alt=""></img>
            </div>
        </div>
    )
}

function Project(props) {
    return (
        <div className="col-xl-6 col-xxl-4  project">
            <div className="card">
                <div id={'project-' + props.id} className="carousel slide" data-interval="false">
                    <ol className="carousel-indicators">
                        {
                            props.project.imgs.map((link, id) =>
                                (<li key={id} data-target={'#project-' + props.id} data-slide-to={id}
                                     className={id === 0 ? 'active' : ''}></li>))
                        }
                    </ol>
                    <div className="carousel-inner">
                        {
                            props.project.imgs.map(
                                (link, id) =>
                                    <ProjectDisplay link={link} key={id} id={id}></ProjectDisplay>)
                        }
                    </div>
                    <a className={'carousel-control-prev'} href={'#project-' + props.id} role="button"
                       data-slide='prev'>
                        <span className={'carousel-control-prev-icon'} aria-hidden="true"></span>
                    </a>
                    <a className={'carousel-control-next'} href={'#project-' + props.id} role="button"
                       data-slide='next'>
                        <span className={'carousel-control-next-icon'} aria-hidden="true"></span>
                    </a>
                </div>
                <div className="card-body">
                    <hr/>
                    <h5 className="card-title">{props.project.title}</h5>
                    {
                        props.project.tags.split(/,\s+/).map(
                            tag => (<div className="tag">{tag}</div>)
                        )
                    }
                    <p className="card-text">{props.project.description}</p>
                    <a target="_blank" href={props.project.link} className="link-btn">{props.project.link}</a>
                </div>
            </div>
        </div>)
}

export function Projects(props) {
    return (
        <div className="row projects">
            {props.projects.map((project, id) => (<Project project={project} id={id} key={id}></Project>))}
        </div>)
}