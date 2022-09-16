export function FrontPage() {
    return (
        <div className="front-page">
            <div className="title">Hi, I'm Feifan</div>
            <div className="text-center">
                {
                    [
                        ['Resume', 'resume.png', '/Feifan He\'s Resume.pdf'],
                        ['LinkedIn', 'linkedin.png', 'https://www.linkedin.com/in/feifanhe'],
                        ['Email', 'email.png', 'mailto:feifanhe@brandeis.edu']
                    ].map((icon, id) => {
                            let [desc, img, href] = icon;
                            return (
                                <a className='icon-container' key={id} href={href} target="_blank">
                                    <img className="icon" src={'./imgs/general/' + img} alt=""></img>
                                    <div className={'icon-desc'}>{desc}</div>
                                </a>
                            )
                        }
                    )
                }
            </div>
        </div>
    )
}