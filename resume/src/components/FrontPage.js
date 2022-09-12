export function FrontPage() {
    return (
        <div className={"front-page"}>
            <div className="title">Hello,<br/>I'm Feifan</div>
            <div className="text-center">
                {
                    [['resume.png', 'Resume'], ['linkedin.png', 'LinkedIn'], ['email.png', 'Email']].map((icon) => {
                            let [img, desc] = icon;
                            return (
                                <div className='icon-container'>
                                    <img className="icon" src={'./imgs/front-page/' + img} alt=""></img>
                                    <div className={'icon-desc'}>{desc}</div>
                                </div>
                            )
                        }
                    )
                }
            </div>
        </div>
    )
}